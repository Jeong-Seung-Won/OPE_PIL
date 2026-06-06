import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Try to import performer, fallback to standard transformer if not available
try:
    from performer_pytorch import Performer
    PERFORMER_AVAILABLE = True
except ImportError:
    print("Warning: performer_pytorch not available. Using standard transformer as fallback.")
    PERFORMER_AVAILABLE = False


class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer('emb', emb)

    def forward(self, x):
        return self.emb[None, :x.shape[1], :].to(x)


class FallbackPerformer(nn.Module):
    def __init__(self, dim, depth, heads, causal=False, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, pos_enc=None, **kwargs):
        # Add positional encoding if provided
        if pos_enc is not None:
            x = x + pos_enc
        
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class PerfPredSqz(nn.Module):
    def __init__(self, Win, Wout, D, heads, dep=8, ff_mult=4):
        super().__init__()
        self.model_name = 'M_seq'
        self.token_emb = nn.Linear(D, D)
        self.pos_enc = FixedPositionalEmbedding(dim=D, max_seq_len=Win)
        self.layer_pos_enc = FixedPositionalEmbedding(dim=D//heads, max_seq_len=Win)
        
        # Calculate a list of dimensions [Win, ... , Wout]
        Ws = np.round(Win * (Wout/Win) ** np.linspace(0, 1, dep+1)).astype(int)
        enc_perf = []
        enc_lin = []
        
        for i in range(dep):
            if PERFORMER_AVAILABLE:
                enc_perf.append(Performer(
                    dim=D, depth=1, heads=heads, causal=False, 
                    feature_redraw_interval=1, dim_head=None
                ))
            else:
                enc_perf.append(FallbackPerformer(
                    dim=D, depth=1, heads=heads, causal=False
                ))
            enc_lin.append(nn.Linear(Ws[i], Ws[i+1]))
            
        self.enc_perf = nn.ModuleList(enc_perf)
        self.enc_lin = nn.ModuleList(enc_lin)
        self.D = D
        
    def forward(self, x, **args):
        for i in range(len(self.enc_perf)):
            if i == 0:
                x = self.token_emb(x) + self.pos_enc(x)[:,:,:x.shape[-1]]
                x = self.enc_perf[0](x, pos_enc=self.layer_pos_enc(x))
            else:
                x = self.enc_perf[i](x)
            x = x.transpose(-1, -2)
            x = self.enc_lin[i](x)
            x = x.transpose(-1, -2)
            if i+1 < len(self.enc_perf):
                x = F.gelu(x)
        x = torch.tanh(x)
        return x


class PerformerAEPositionalEncoding(nn.Module):
    def __init__(self, W, D, heads, ff_mult=4, dep=4, lat=10, c1={'out':40,'kern':6,'strd':2}, return_lat=False):
        super().__init__()
        self.model_name = 'M_pt'
        self.token_emb = nn.Linear(D, D)
        
        if PERFORMER_AVAILABLE:
            self.enc_perf = Performer(
                dim=D, depth=dep, heads=heads, causal=False, 
                feature_redraw_interval=1, dim_head=None
            )
            self.dec_perf = Performer(
                dim=D, depth=dep, heads=heads, causal=False, 
                feature_redraw_interval=1, dim_head=None
            )
        else:
            self.enc_perf = FallbackPerformer(
                dim=D, depth=dep, heads=heads, causal=False
            )
            self.dec_perf = FallbackPerformer(
                dim=D, depth=dep, heads=heads, causal=False
            )
        
        self.enc_lin = nn.Sequential(nn.Linear(D, lat), nn.GELU())
        self.dec_lin = nn.Linear(lat, D)
        
        self.W = W
        self.D = D
        self.lat = lat
        self.return_lat = return_lat
        self.pos_enc = FixedPositionalEmbedding(dim=D, max_seq_len=W)
        self.layer_pos_enc = FixedPositionalEmbedding(dim=D//heads, max_seq_len=W)
        
    def forward(self, x, **args):
        x = self.token_emb(x) + self.pos_enc(x)[:,:,:x.shape[-1]]
        x = self.enc_perf(x, pos_enc=self.layer_pos_enc(x))
        z = self.enc_lin(x)
        
        x = self.dec_lin(z)
        x = self.dec_perf(x, pos_enc=self.layer_pos_enc(x))
        x = torch.tanh(x)
        return [z, x] if self.return_lat else x


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        
        # Model parameters
        self.seq_len = args.seq_len
        self.enc_in = getattr(args, 'enc_in', 6)
        
        # NPSR specific parameters
        self.npsr_type = getattr(args, 'npsr_type', 'autoencoder')  # 'autoencoder' or 'squeezing'
        self.d_model = getattr(args, 'npsr_d_model', 64)
        self.n_heads = getattr(args, 'npsr_n_heads', 8)
        self.depth = getattr(args, 'npsr_depth', 4)
        self.latent_dim = getattr(args, 'npsr_latent_dim', 10)
        self.ff_mult = getattr(args, 'npsr_ff_mult', 4)
        
        # Input/output projection layers
        self.input_projection = nn.Linear(self.enc_in, self.d_model)
        self.output_projection = nn.Linear(self.d_model, self.enc_in)
        
        # Choose model type
        if self.npsr_type == 'squeezing':
            # For squeezing model, we need output length (using same as input for reconstruction)
            self.npsr_model = PerfPredSqz(
                Win=self.seq_len,
                Wout=self.seq_len,  # Same length for reconstruction
                D=self.d_model,
                heads=self.n_heads,
                dep=self.depth,
                ff_mult=self.ff_mult
            )
        else:  # autoencoder
            self.npsr_model = PerformerAEPositionalEncoding(
                W=self.seq_len,
                D=self.d_model,
                heads=self.n_heads,
                ff_mult=self.ff_mult,
                dep=self.depth,
                lat=self.latent_dim,
                return_lat=False
            )
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def encode(self, x, x_mark=None, y_mark=None):
        # Project to model dimension
        x_proj = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        if self.npsr_type == 'autoencoder':
            # For autoencoder, we can get latent representation
            self.npsr_model.return_lat = True
            z, _ = self.npsr_model(x_proj)
            self.npsr_model.return_lat = False
            return z
        else:
            # For squeezing model, return the intermediate representation
            return self.npsr_model(x_proj)
    
    def forward(self, x, x_mark=None, y_mark=None):
        # Project to model dimension
        x_proj = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        # Pass through NPSR model
        if self.npsr_type == 'autoencoder':
            reconstructed_proj = self.npsr_model(x_proj)
        else:  # squeezing
            reconstructed_proj = self.npsr_model(x_proj)
        
        # Project back to original feature space
        reconstructed = self.output_projection(reconstructed_proj)
        
        # Store for potential loss computation
        if self.training:
            self.last_input = x
            self.last_reconstruction = reconstructed
        
        return reconstructed
    
    def compute_npsr_loss(self):
        if not hasattr(self, 'last_input'):
            return {'total_loss': torch.tensor(0.0)}
        
        # Basic reconstruction loss
        recon_loss = F.mse_loss(self.last_reconstruction, self.last_input)
        
        return {
            'total_loss': recon_loss,
            'recon_loss': recon_loss.item()
        }
    
    def compute_anomaly_score(self, x):
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            
            # Compute reconstruction error per time step
            scores = torch.mean((x - reconstructed) ** 2, dim=2)  # [batch_size, seq_len]
            
            return scores