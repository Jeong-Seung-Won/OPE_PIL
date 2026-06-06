import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import MultivariateNormal


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        
        # Model parameters
        self.seq_len = args.seq_len
        self.enc_in = getattr(args, 'enc_in', 6)
        self.z_dim = getattr(args, 'z_dim', 16)  # Latent dimension
        self.n_gmm = getattr(args, 'n_gmm', 4)   # Number of Gaussian components
        
        # Input flattening dimension
        self.input_dim = self.seq_len * self.enc_in
        
        # Hidden dimensions
        self.hidden_dim1 = getattr(args, 'hidden_dim1', 128)
        self.hidden_dim2 = getattr(args, 'hidden_dim2', 64)
        self.hidden_dim3 = getattr(args, 'hidden_dim3', 32)
        
        # Encoder network (Compression network)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim1),
            nn.Tanh(),
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            nn.Tanh(),
            nn.Linear(self.hidden_dim2, self.hidden_dim3),
            nn.Tanh(),
            nn.Linear(self.hidden_dim3, self.z_dim)
        )
        
        # Decoder network (Reconstruction network)
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, self.hidden_dim3),
            nn.Tanh(),
            nn.Linear(self.hidden_dim3, self.hidden_dim2),
            nn.Tanh(),
            nn.Linear(self.hidden_dim2, self.hidden_dim1),
            nn.Tanh(),
            nn.Linear(self.hidden_dim1, self.input_dim)
        )
        
        # Estimation network (GMM parameter estimation)
        # Input: z_c (latent) + reconstruction features (2D)
        estimation_input_dim = self.z_dim + 2
        self.estimation_network = nn.Sequential(
            nn.Linear(estimation_input_dim, 16),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(16, self.n_gmm),
            nn.Softmax(dim=1)
        )
        
        # GMM parameters (learnable)
        self.phi = nn.Parameter(torch.ones(self.n_gmm) / self.n_gmm)  # Mixture weights
        self.mu = nn.Parameter(torch.randn(self.n_gmm, estimation_input_dim))  # Means
        self.cov = nn.Parameter(torch.eye(estimation_input_dim).unsqueeze(0).repeat(self.n_gmm, 1, 1))  # Covariances
        
        # Training phase tracking
        self.lambda_energy = getattr(args, 'lambda_energy', 0.1)
        self.lambda_cov = getattr(args, 'lambda_cov', 0.005)
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode(self, x, x_mark=None, y_mark=None):
        # Flatten input
        x_flat = x.view(x.size(0), -1)  # [batch_size, seq_len * n_features]
        
        # Encode
        z_c = self.encoder(x_flat)
        
        return z_c
    
    def decode(self, z_c):
        return self.decoder(z_c)
    
    def compute_reconstruction_features(self, x_flat, x_hat_flat):
        # Relative Euclidean distance
        euclidean_dist = torch.norm(x_flat - x_hat_flat, p=2, dim=1)
        x_norm = torch.norm(x_flat, p=2, dim=1)
        rec_euclidean = (euclidean_dist / (x_norm + 1e-8)).unsqueeze(1)
        
        # Cosine similarity
        rec_cosine = F.cosine_similarity(x_flat, x_hat_flat, dim=1).unsqueeze(1)
        
        return rec_euclidean, rec_cosine
    
    def forward(self, x, x_mark=None, y_mark=None):
        batch_size = x.size(0)
        
        # Flatten input
        x_flat = x.view(batch_size, -1)
        
        # Encode
        z_c = self.encoder(x_flat)
        
        # Decode
        x_hat_flat = self.decoder(z_c)
        
        # Compute reconstruction features
        rec_euclidean, rec_cosine = self.compute_reconstruction_features(x_flat, x_hat_flat)
        
        # Augmented latent representation
        z = torch.cat([z_c, rec_euclidean, rec_cosine], dim=1)
        
        # Estimate mixture membership
        gamma = self.estimation_network(z)
        
        # Store for loss computation
        if self.training:
            self.last_z_c = z_c
            self.last_z = z
            self.last_gamma = gamma
            self.last_x_flat = x_flat
            self.last_x_hat_flat = x_hat_flat
        
        # Reshape for compatibility with existing framework
        x_reconstructed = x_hat_flat.view(batch_size, self.seq_len, self.enc_in)
        
        return x_reconstructed
    
    def compute_energy(self, z, gamma):
        batch_size = z.size(0)
        
        # Regularize covariance matrices (ensure positive definiteness)
        cov_reg = self.cov + torch.eye(z.size(1)).unsqueeze(0).to(z.device) * 1e-6
        
        # Compute energy for each component
        energy_components = []
        
        for k in range(self.n_gmm):
            # Multivariate normal distribution
            try:
                mvn = MultivariateNormal(self.mu[k], cov_reg[k])
                log_prob = mvn.log_prob(z)  # [batch_size]
            except:
                # Fallback to simple computation if covariance is problematic
                diff = z - self.mu[k].unsqueeze(0)  # [batch_size, z_dim+2]
                try:
                    inv_cov = torch.inverse(cov_reg[k])
                    mahalanobis = torch.sum(diff * torch.matmul(diff, inv_cov), dim=1)
                    log_prob = -0.5 * mahalanobis
                except:
                    # Simple Euclidean distance fallback
                    log_prob = -torch.sum(diff ** 2, dim=1)
            
            # Add mixture weight
            energy_k = gamma[:, k] * torch.exp(log_prob + torch.log(self.phi[k] + 1e-8))
            energy_components.append(energy_k)
        
        # Sum over components and take negative log
        total_energy = torch.stack(energy_components, dim=1).sum(dim=1)
        energy = -torch.log(total_energy + 1e-8)
        
        return energy
    
    def compute_dagmm_loss(self):
        if not hasattr(self, 'last_z_c'):
            return {'total_loss': torch.tensor(0.0)}
        
        # Reconstruction loss
        recon_loss = F.mse_loss(self.last_x_hat_flat, self.last_x_flat)
        
        # Energy loss (regularization term)
        energy = self.compute_energy(self.last_z, self.last_gamma)
        energy_loss = torch.mean(energy)
        
        # Covariance regularization (encourage diversity)
        cov_loss = 0.0
        for k in range(self.n_gmm):
            cov_k = self.cov[k]
            # Penalize small eigenvalues
            eigenvals = torch.diagonal(cov_k, dim1=-2, dim2=-1)
            cov_loss += torch.sum(1.0 / (eigenvals + 1e-8))
        cov_loss = cov_loss / self.n_gmm
        
        # Total loss
        total_loss = recon_loss + self.lambda_energy * energy_loss + self.lambda_cov * cov_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss.item(),
            'energy_loss': energy_loss.item(),
            'cov_loss': cov_loss if isinstance(cov_loss, float) else cov_loss.item(),
            'lambda_energy': self.lambda_energy,
            'lambda_cov': self.lambda_cov
        }
    
    def compute_anomaly_score(self, x):
        self.eval()
        with torch.no_grad():
            batch_size = x.size(0)
            
            # Flatten and process
            x_flat = x.view(batch_size, -1)
            z_c = self.encoder(x_flat)
            x_hat_flat = self.decoder(z_c)
            
            # Reconstruction features
            rec_euclidean, rec_cosine = self.compute_reconstruction_features(x_flat, x_hat_flat)
            z = torch.cat([z_c, rec_euclidean, rec_cosine], dim=1)
            gamma = self.estimation_network(z)
            
            # Compute energy (anomaly score)
            energy = self.compute_energy(z, gamma)  # [batch_size]
            
            # Expand to match expected output shape [batch_size, seq_len]
            # For DAGMM, we assign same score to all time points in sequence
            scores = energy.unsqueeze(1).repeat(1, self.seq_len)
            
            return scores
    
    def get_gmm_parameters(self):
        return {
            'phi': self.phi.data.cpu().numpy(),
            'mu': self.mu.data.cpu().numpy(),
            'cov': self.cov.data.cpu().numpy()
        }