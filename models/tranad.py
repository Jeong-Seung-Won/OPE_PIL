import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TranADTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, seq_len):
        super(TranADTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=False  # TranAD uses seq_len first format
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, input_dim)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, input_dim]
        
        # Input projection
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)
        
        # Output projection
        output = self.output_projection(encoded)
        
        # Back to batch first
        output = output.permute(1, 0, 2)  # [batch_size, seq_len, input_dim]
        
        return output


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        
        # Model parameters
        self.seq_len = args.seq_len
        self.enc_in = getattr(args, 'enc_in', 6)
        self.d_model = getattr(args, 'd_model', 128)
        self.nhead = getattr(args, 'n_heads', 8)
        self.num_layers = getattr(args, 'n_layers', 3)
        
        # TranAD specific parameters
        self.k = getattr(args, 'tranad_k', 3)  # Training phase switch parameter
        
        # Dual transformer architecture
        self.transformer1 = TranADTransformer(
            self.enc_in, self.d_model, self.nhead, self.num_layers, self.seq_len
        )
        self.transformer2 = TranADTransformer(
            self.enc_in, self.d_model, self.nhead, self.num_layers, self.seq_len
        )
        
        # Training phase tracking
        self.training_phase = 1
        self.epoch_counter = 0
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.xavier_uniform_(module.out_proj.weight)
    
    def encode(self, x, x_mark=None, y_mark=None):
        # Use first transformer's intermediate representation
        x_proj = x.permute(1, 0, 2)  # [seq_len, batch_size, input_dim]
        x_proj = self.transformer1.input_projection(x_proj) * math.sqrt(self.d_model)
        x_proj = self.transformer1.pos_encoder(x_proj)
        encoded = self.transformer1.transformer_encoder(x_proj)
        
        return encoded.permute(1, 0, 2)  # [batch_size, seq_len, d_model]
    
    def forward(self, x, x_mark=None, y_mark=None):
        if self.training_phase == 1:
            # Phase 1: Standard autoencoder training
            # Both transformers learn to reconstruct independently
            recon1 = self.transformer1(x)
            recon2 = self.transformer2(x)
            
            # Store for loss computation
            if self.training:
                self.last_recon1 = recon1
                self.last_recon2 = recon2
                self.last_input = x
            
            # Return average for consistent interface
            return (recon1 + recon2) / 2
            
        else:
            # Phase 2: Adversarial training
            # T1 reconstructs input, T2 reconstructs T1's output
            recon1 = self.transformer1(x)
            recon2 = self.transformer2(recon1)
            
            # Store for loss computation
            if self.training:
                self.last_recon1 = recon1
                self.last_recon2 = recon2
                self.last_input = x
            
            return recon1  # Primary reconstruction from T1
    
    def compute_tranad_loss(self):
        if not hasattr(self, 'last_input'):
            return {'total_loss': torch.tensor(0.0)}
        
        x = self.last_input
        recon1 = self.last_recon1
        recon2 = self.last_recon2
        
        if self.training_phase == 1:
            # Phase 1: Independent reconstruction loss
            loss1 = F.mse_loss(recon1, x)
            loss2 = F.mse_loss(recon2, x)
            total_loss = (loss1 + loss2) / 2
            
            return {
                'total_loss': total_loss,
                'loss1': loss1.item(),
                'loss2': loss2.item(),
                'phase': 1
            }
        else:
            # Phase 2: Adversarial training
            # L1: T1 reconstruction loss
            loss1 = F.mse_loss(recon1, x)
            
            # L2: T2 reconstruction loss (T2 tries to reconstruct T1's output)
            loss2 = F.mse_loss(recon2, x)
            
            # Combined loss with focusing mechanism
            combined_loss = (1 / self.k) * loss1 + (1 - 1/self.k) * loss2
            
            return {
                'total_loss': combined_loss,
                'loss1': loss1.item(),
                'loss2': loss2.item(),
                'combined_loss': combined_loss.item(),
                'phase': 2
            }
    
    def update_training_phase(self, epoch):
        phase_switch_epoch = getattr(self.args, 'tranad_phase_switch', 25)
        if epoch >= phase_switch_epoch and self.training_phase == 1:
            self.training_phase = 2
            print(f"TranAD: Switching to adversarial training phase at epoch {epoch}")
    
    def compute_anomaly_score(self, x):
        self.eval()
        with torch.no_grad():
            # Get reconstructions from both transformers
            if self.training_phase == 1:
                recon1 = self.transformer1(x)
                recon2 = self.transformer2(x)
                
                # Combine reconstruction errors
                error1 = torch.mean((x - recon1) ** 2, dim=2)
                error2 = torch.mean((x - recon2) ** 2, dim=2)
                scores = (error1 + error2) / 2
                
            else:
                recon1 = self.transformer1(x)
                recon2 = self.transformer2(recon1)
                
                # TranAD score: focus on T1's reconstruction
                error1 = torch.mean((x - recon1) ** 2, dim=2)
                error2 = torch.mean((x - recon2) ** 2, dim=2)
                
                # Weighted combination (more weight on T1 in phase 2)
                scores = (1/self.k) * error1 + (1 - 1/self.k) * error2
            
            return scores
    
    def get_training_info(self):
        return {
            'training_phase': self.training_phase,
            'k_parameter': self.k,
            'phase_switch_epoch': getattr(self.args, 'tranad_phase_switch', 25)
        }