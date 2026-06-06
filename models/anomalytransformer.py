import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
    
    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class AnomalyAttention(nn.Module):
    def __init__(self, win_size, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
        # Distance matrix for prior computation
        self.win_size = win_size
        self.register_buffer('distances', self._create_distance_matrix(win_size))
    
    def _create_distance_matrix(self, win_size):
        distances = torch.zeros((win_size, win_size))
        for i in range(win_size):
            for j in range(win_size):
                distances[i][j] = abs(i - j)
        return distances
    
    def forward(self, queries, keys, values, sigma, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)
        
        # Compute attention scores
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -np.inf)
        
        attn = scale * scores
        
        # Process sigma for prior computation
        sigma = sigma.transpose(1, 2)  # B L H -> B H L
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, self.win_size)  # B H L L
        
        # Compute Gaussian prior based on distances
        distances = self.distances.unsqueeze(0).unsqueeze(0).repeat(
            sigma.shape[0], sigma.shape[1], 1, 1
        ).to(sigma.device)
        
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(
            -distances ** 2 / (2 * sigma ** 2)
        )
        
        # Compute series (attention weights)
        series = self.dropout(torch.softmax(attn, dim=-1))
        
        # Apply attention to values
        V = torch.einsum("bhls,bshd->blhd", series, values)
        
        if self.output_attention:
            return V.contiguous(), series, prior, sigma
        else:
            return V.contiguous(), None, None, None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
    
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        # Store input for residual connection
        x = queries
        
        # Project to multi-head
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)
        
        # Apply attention
        out, series, prior, sigma = self.inner_attention(
            queries, keys, values, sigma, attn_mask
        )
        
        # Project back
        out = out.view(B, L, -1)
        return self.out_projection(out), series, prior, sigma


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    
    def forward(self, x, attn_mask=None):
        # Self-attention
        new_x, attn, mask, sigma = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        
        # Feed-forward network
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
    
    def forward(self, x, attn_mask=None):
        # Store attention outputs from all layers
        series_list = []
        prior_list = []
        sigma_list = []
        
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x, series_list, prior_list, sigma_list


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        
        # Model parameters
        self.seq_len = args.seq_len
        self.enc_in = getattr(args, 'enc_in', 6)
        self.d_model = getattr(args, 'd_model', 512)
        self.n_heads = getattr(args, 'n_heads', 8)
        self.e_layers = getattr(args, 'e_layers', 3)
        self.d_ff = getattr(args, 'd_ff', 512)
        self.dropout = getattr(args, 'dropout', 0.0)
        self.activation = getattr(args, 'activation', 'gelu')
        self.output_attention = getattr(args, 'output_attention', True)
        
        # Association discrepancy parameters
        self.lambda_association = getattr(args, 'lambda_association', 1.0)
        
        # Embedding
        self.embedding = DataEmbedding(self.enc_in, self.d_model, self.dropout)
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(
                            self.seq_len,
                            attention_dropout=self.dropout,
                            output_attention=self.output_attention
                        ),
                        self.d_model, self.n_heads
                    ),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=nn.LayerNorm(self.d_model)
        )
        
        # Output projection
        self.projection = nn.Linear(self.d_model, self.enc_in, bias=True)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
    
    def encode(self, x, x_mark=None, y_mark=None):
        # Embedding
        enc_out = self.embedding(x)
        
        # Encoding
        enc_out, _, _, _ = self.encoder(enc_out)
        
        return enc_out
    
    def forward(self, x, x_mark=None, y_mark=None):
        # Embedding
        enc_out = self.embedding(x)
        
        # Encoding
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        
        # Projection to output space
        enc_out = self.projection(enc_out)
        
        # Store attention information for loss computation
        if self.training:
            self.last_series = series
            self.last_prior = prior
            self.last_sigmas = sigmas
            self.last_input = x
            self.last_reconstruction = enc_out
        
        return enc_out
    
    def compute_association_discrepancy_loss(self):
        if not hasattr(self, 'last_series'):
            return {'total_loss': torch.tensor(0.0)}
        
        # Reconstruction loss
        recon_loss = F.mse_loss(self.last_reconstruction, self.last_input)
        
        # Association discrepancy loss
        association_loss = 0.0
        
        for layer_idx, (series, prior) in enumerate(zip(self.last_series, self.last_prior)):
            if series is not None and prior is not None:
                # Compute KL divergence between series and prior
                # series: [B, H, L, L], prior: [B, H, L, L]
                kl_loss = F.kl_div(
                    F.log_softmax(series, dim=-1),
                    F.softmax(prior, dim=-1),
                    reduction='batchmean'
                )
                association_loss += kl_loss
        
        association_loss = association_loss / len(self.last_series)
        
        # Total loss
        total_loss = recon_loss + self.lambda_association * association_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss.item(),
            'association_loss': association_loss.item() if isinstance(association_loss, torch.Tensor) else association_loss,
            'lambda_association': self.lambda_association
        }
    
    def compute_anomaly_score(self, x):
        self.eval()
        with torch.no_grad():
            # Forward pass
            enc_out = self.embedding(x)
            enc_out, series_list, prior_list, _ = self.encoder(enc_out)
            reconstructed = self.projection(enc_out)
            
            # Reconstruction error
            recon_error = torch.mean((x - reconstructed) ** 2, dim=2)  # [B, L]
            
            # Association discrepancy score
            association_scores = []
            
            for series, prior in zip(series_list, prior_list):
                if series is not None and prior is not None:
                    # Compute discrepancy between series and prior for each time step
                    # Average over heads and sequence positions
                    discrepancy = torch.mean(
                        F.kl_div(
                            F.log_softmax(series, dim=-1),
                            F.softmax(prior, dim=-1),
                            reduction='none'
                        ),
                        dim=(1, 3)  # Average over heads and target positions
                    )  # [B, L]
                    association_scores.append(discrepancy)
            
            # Combine association scores from all layers
            if association_scores:
                avg_association_score = torch.stack(association_scores).mean(dim=0)
            else:
                avg_association_score = torch.zeros_like(recon_error)
            
            # Combine reconstruction error and association discrepancy
            combined_scores = 0.5 * recon_error + 0.5 * avg_association_score
            
            return combined_scores
    
    def get_attention_info(self):
        if hasattr(self, 'last_series'):
            return {
                'series': [s.detach().cpu().numpy() if s is not None else None for s in self.last_series],
                'prior': [p.detach().cpu().numpy() if p is not None else None for p in self.last_prior],
                'sigmas': [s.detach().cpu().numpy() if s is not None else None for s in self.last_sigmas]
            }
        return None