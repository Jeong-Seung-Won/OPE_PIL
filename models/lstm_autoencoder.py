"""
LSTM AutoEncoder model for anomaly detection
"""

import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):
    """
    LSTM-based AutoEncoder for time series anomaly detection
    This class follows the pattern expected by exp_basic.py
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        
        # Model parameters
        self.seq_len = args.seq_len
        self.enc_in = getattr(args, 'enc_in', 6)
        self.d_model = getattr(args, 'd_model', 128)
        self.n_layers = getattr(args, 'n_layers', 2)
        self.dropout = getattr(args, 'dropout', 0.1)
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=self.d_model,
            num_layers=self.n_layers,
            dropout=self.dropout if self.n_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Latent representation layer
        self.latent_projection = nn.Linear(self.d_model, self.d_model // 2)
        
        # Decoder initialization layer
        self.decoder_init = nn.Linear(self.d_model // 2, self.d_model)
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=self.n_layers,
            dropout=self.dropout if self.n_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Output projection
        self.output_projection = nn.Linear(self.d_model, self.enc_in)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                # Set forget gate bias to 1
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.)
    
    def encode(self, x, x_mark=None, y_mark=None):
        """
        Extract latent representation for anomaly detection
        Args:
            x: [batch_size, seq_len, n_features]
        Returns:
            latent: [batch_size, latent_dim]
        """
        # Encode sequence
        encoded, (hidden, cell) = self.encoder_lstm(x)
        
        # Use last hidden state as sequence representation
        sequence_repr = hidden[-1]  # [batch_size, d_model]
        
        # Project to latent space
        latent = self.latent_projection(sequence_repr)  # [batch_size, latent_dim]
        return latent
    
    def forward(self, x, x_mark=None, y_mark=None):
        """
        Forward pass for reconstruction
        Args:
            x: [batch_size, seq_len, n_features]
        Returns:
            reconstructed: [batch_size, seq_len, n_features]
        """
        batch_size, seq_len, _ = x.shape
        
        # Encode
        encoded, (hidden, cell) = self.encoder_lstm(x)
        
        # Get latent representation
        sequence_repr = hidden[-1]  # [batch_size, d_model]
        latent = self.latent_projection(sequence_repr)
        
        # Initialize decoder hidden state
        decoder_hidden = self.decoder_init(latent)  # [batch_size, d_model]
        decoder_hidden = decoder_hidden.unsqueeze(0).repeat(self.n_layers, 1, 1)  # [n_layers, batch_size, d_model]
        decoder_cell = torch.zeros_like(decoder_hidden)
        
        # Prepare decoder input (start with latent representation repeated)
        decoder_input = latent.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, latent_dim]
        decoder_input = self.decoder_init(decoder_input)  # [batch_size, seq_len, d_model]
        
        # Decode
        decoded, _ = self.decoder_lstm(decoder_input, (decoder_hidden, decoder_cell))
        
        # Project to original feature space
        reconstructed = self.output_projection(decoded)
        
        return reconstructed