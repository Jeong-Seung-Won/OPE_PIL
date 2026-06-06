import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Ensure d_model is not None and is a valid integer
        if d_model is None:
            raise ValueError("d_model cannot be None")
        
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        if d_model is None:
            raise ValueError("d_model cannot be None")
        if n_heads is None:
            n_heads = 8  # Default value
            
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        seq_len = Q.size(1)
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        return context, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Linear transformations and split into heads
        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        context, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Final linear transformation
        output = self.W_o(context)
        
        return output, attention_weights


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        
        if d_model is None:
            raise ValueError("d_model cannot be None")
        if d_ff is None:
            d_ff = d_model * 4  # Default value
            
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        
        # Basic parameters
        self.seq_len = args.seq_len
        self.enc_in = getattr(args, 'enc_in', 6)
        
        # DTAAD specific parameters with proper defaults
        self.window_size = getattr(args, 'dtaad_window_size', 10)
        
        # Set d_model with proper fallback
        self.d_model = getattr(args, 'dtaad_d_model', None)
        if self.d_model is None:
            self.d_model = self.enc_in * 16  # Default: 16 times the input dimension
        
        # Set n_heads with proper fallback
        self.n_heads = getattr(args, 'dtaad_n_heads', None)
        if self.n_heads is None:
            # Ensure n_heads divides d_model evenly
            if self.d_model >= 8 and self.d_model % 8 == 0:
                self.n_heads = 8
            elif self.d_model >= 4 and self.d_model % 4 == 0:
                self.n_heads = 4
            elif self.d_model >= 2 and self.d_model % 2 == 0:
                self.n_heads = 2
            else:
                self.n_heads = 1
        
        self.d_ff = getattr(args, 'dtaad_d_ff', self.d_model * 4)
        self.n_layers = getattr(args, 'n_layers', 2)
        self.dropout = getattr(args, 'dropout', 0.1)
        self.lambda_param = getattr(args, 'dtaad_lambda', 0.8)
        
        # Validate parameters
        if self.d_model % self.n_heads != 0:
            # Adjust n_heads to be compatible with d_model
            for heads in [8, 4, 2, 1]:
                if self.d_model % heads == 0:
                    self.n_heads = heads
                    break
        
        print(f"DTAAD Model initialized with:")
        print(f"  d_model: {self.d_model}")
        print(f"  n_heads: {self.n_heads}")
        print(f"  d_ff: {self.d_ff}")
        print(f"  n_layers: {self.n_layers}")
        print(f"  window_size: {self.window_size}")
        
        # Input embedding
        self.input_embedding = nn.Linear(self.enc_in, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout, max_len=self.seq_len)
        
        # Transformer encoder
        self.transformer_encoder = TransformerEncoder(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            n_layers=self.n_layers,
            dropout=self.dropout
        )
        
        # Output layers
        self.output_projection = nn.Linear(self.d_model, self.enc_in)
        
        # Additional layers for anomaly detection
        self.anomaly_detector = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 1),
            nn.Sigmoid()
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
        # Input embedding
        x = self.input_embedding(x)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)
        
        return encoded
    
    def forward(self, x, x_mark=None, y_mark=None):
        # Encode
        encoded = self.encode(x, x_mark, y_mark)
        
        # Reconstruct
        reconstructed = self.output_projection(encoded)
        
        # Store for potential loss computation
        if self.training:
            self.last_encoded = encoded
            self.last_input = x
            self.last_reconstruction = reconstructed
        
        return reconstructed
    
    def compute_dtaad_loss(self):
        if not hasattr(self, 'last_input'):
            return {'total_loss': torch.tensor(0.0)}
        
        # Reconstruction loss
        recon_loss = F.mse_loss(self.last_reconstruction, self.last_input)
        
        # Anomaly detection loss (optional, can be used during training)
        anomaly_scores = self.anomaly_detector(self.last_encoded)
        # For unsupervised learning, we can use reconstruction error as pseudo-labels
        pseudo_labels = torch.mean((self.last_reconstruction - self.last_input) ** 2, dim=2, keepdim=True)
        pseudo_labels = (pseudo_labels > torch.median(pseudo_labels)).float()
        
        anomaly_loss = F.binary_cross_entropy(anomaly_scores, pseudo_labels)
        
        # Combined loss
        total_loss = recon_loss + self.lambda_param * anomaly_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss.item(),
            'anomaly_loss': anomaly_loss.item(),
            'lambda_param': self.lambda_param
        }
    
    def compute_anomaly_score(self, x):
        self.eval()
        with torch.no_grad():
            # Get reconstruction
            reconstructed = self.forward(x)
            
            # Reconstruction error
            recon_error = torch.mean((x - reconstructed) ** 2, dim=2)  # [batch_size, seq_len]
            
            # Learned anomaly scores
            encoded = self.encode(x)
            learned_scores = self.anomaly_detector(encoded).squeeze(-1)  # [batch_size, seq_len]
            
            # Combine scores
            combined_scores = 0.5 * recon_error + 0.5 * learned_scores
            
            return combined_scores