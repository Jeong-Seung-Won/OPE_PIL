import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
from sklearn.cluster import KMeans


# Attention Layer
class Attention(nn.Module):
    def __init__(self, window_size, mask_flag=False, scale=None, dropout=0.0):
        super(Attention, self).__init__()
        self.window_size = window_size
        self.mask_flag = mask_flag
        self.scale = scale
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        N, L, Head, C = queries.shape
        scale = self.scale if self.scale is not None else 1. / math.sqrt(C)
        attn_scores = torch.einsum('nlhd,nshd->nhls', queries, keys)
        attn_weights = self.dropout(torch.softmax(scale * attn_scores, dim=-1))
        updated_values = torch.einsum('nhls,nshd->nlhd', attn_weights, values)
        return updated_values.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, window_size, d_model, n_heads, d_keys=None, d_values=None, mask_flag=False, 
                 scale=None, dropout=0.0):
        super(AttentionLayer, self).__init__()
        self.d_keys = d_keys if d_keys is not None else (d_model // n_heads)
        self.d_values = d_values if d_values is not None else (d_model // n_heads)
        self.n_heads = n_heads
        self.d_model = d_model

        self.W_Q = nn.Linear(self.d_model, self.n_heads * self.d_keys)
        self.W_K = nn.Linear(self.d_model, self.n_heads * self.d_keys)
        self.W_V = nn.Linear(self.d_model, self.n_heads * self.d_values)
        self.out_proj = nn.Linear(self.n_heads * self.d_values, self.d_model)
        self.attn = Attention(window_size=window_size, mask_flag=mask_flag, scale=scale, dropout=dropout)

    def forward(self, input):
        N, L, _ = input.shape
        Q = self.W_Q(input).contiguous().view(N, L, self.n_heads, -1)
        K = self.W_K(input).contiguous().view(N, L, self.n_heads, -1)
        V = self.W_V(input).contiguous().view(N, L, self.n_heads, -1)
        updated_V = self.attn(Q, K, V)
        out = updated_V.view(N, L, -1)
        return self.out_proj(out)


# Embedding Layers
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.pe = torch.zeros((max_len, d_model), dtype=torch.float)
        self.pe.requires_grad = False

        pos = torch.arange(0, max_len).float().unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2).float()

        self.pe[:, ::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, in_dim, d_model):
        super(TokenEmbedding, self).__init__()
        pad = 1 if torch.__version__ >= '1.5.0' else 2
        self.conv = nn.Conv1d(in_channels=in_dim, out_channels=d_model, kernel_size=3, padding=pad, 
                              padding_mode='circular', bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class InputEmbedding(nn.Module):
    def __init__(self, in_dim, d_model, device, dropout=0.0):
        super(InputEmbedding, self).__init__()
        self.device = device
        self.token_embedding = TokenEmbedding(in_dim=in_dim, d_model=d_model)
        self.pos_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        pos_emb = self.pos_embedding(x)
        if x.is_cuda:
            pos_emb = pos_emb.cuda()
        x = self.token_embedding(x) + pos_emb.to(x.device)
        return self.dropout(x)


# Memory Module
class MemoryModule(nn.Module):
    def __init__(self, n_memory, fea_dim, shrink_thres=0.0025, device=None, memory_init_embedding=None, phase_type=None, dataset_name=None):
        super(MemoryModule, self).__init__()
        self.n_memory = n_memory
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.device = device
        self.phase_type = phase_type
        self.memory_init_embedding = memory_init_embedding
        
        self.U = nn.Linear(fea_dim, fea_dim)
        self.W = nn.Linear(fea_dim, fea_dim)
        
        # Initialize memory
        if self.memory_init_embedding is None:
            if self.phase_type == 'test':
                # Create memory_item directory if it doesn't exist
                os.makedirs('./memory_item', exist_ok=True)
                load_path = f'./memory_item/{dataset_name}_memory_item.pth'
                if os.path.exists(load_path):
                    self.mem = torch.load(load_path)
                    print(f'Loading memory item vectors from {load_path}')
                else:
                    print(f'Memory file {load_path} not found, using random initialization')
                    self.mem = F.normalize(torch.rand((self.n_memory, self.fea_dim), dtype=torch.float), dim=1)
            else:
                print('Loading memory item with random initialization (for first train phase)')
                self.mem = F.normalize(torch.rand((self.n_memory, self.fea_dim), dtype=torch.float), dim=1)
        else:
            if self.phase_type == 'second_train':
                print('Second training phase with initialized memory')
                self.mem = memory_init_embedding

    def hard_shrink_relu(self, input, lambd=0.0025, epsilon=1e-12):
        output = (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)
        return output
    
    def get_attn_score(self, query, key):
        attn = torch.matmul(query, torch.t(key.to(query.device)))
        attn = F.softmax(attn, dim=-1)

        if self.shrink_thres > 0:
            attn = self.hard_shrink_relu(attn, self.shrink_thres)
            attn = F.normalize(attn, p=1, dim=1)
        
        return attn
    
    def read(self, query):
        self.mem = self.mem.to(query.device)
        attn = self.get_attn_score(query, self.mem.detach())
        add_memory = torch.matmul(attn, self.mem.detach())
        read_query = torch.cat((query, add_memory), dim=1)
        return {'output': read_query, 'attn': attn}

    def update(self, query):
        self.mem = self.mem.to(query.device)
        attn = self.get_attn_score(self.mem, query.detach())
        add_mem = torch.matmul(attn, query.detach())
        update_gate = torch.sigmoid(self.U(self.mem) + self.W(add_mem))
        self.mem = (1 - update_gate) * self.mem + update_gate * add_mem

    def forward(self, query):
        s = query.data.shape
        l = len(s)

        query = query.contiguous()
        query = query.view(-1, s[-1])

        if self.phase_type != 'test':
            self.update(query)
        
        outs = self.read(query)
        read_query, attn = outs['output'], outs['attn']
        
        if l == 2:
            pass
        elif l == 3:
            read_query = read_query.view(s[0], s[1], 2*s[2])
            attn = attn.view(s[0], s[1], self.n_memory)
        else:
            raise TypeError('Wrong input dimension')
        
        return {'output': read_query, 'attn': attn, 'memory_init_embedding': self.mem}


# Transformer Components
class EncoderLayer(nn.Module):
    def __init__(self, attn, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.attn_layer = attn
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x):
        out = self.attn_layer(x)
        x = x + self.dropout(out)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        for attn_layer in self.attn_layers:
            x = attn_layer(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, c_out, d_ff=None, activation='relu', dropout=0.1):
        super(Decoder, self).__init__()
        self.out_linear = nn.Linear(d_model, c_out)

    def forward(self, x):
        return self.out_linear(x)


# Main MEMTO Model
class TransformerVar(nn.Module):
    def __init__(self, win_size, enc_in, c_out, n_memory, shrink_thres=0.0025, 
                 d_model=512, n_heads=8, e_layers=3, d_ff=512, dropout=0.0, activation='gelu', 
                 device=None, memory_init_embedding=None, memory_initial=False, phase_type=None, dataset_name=None):
        super(TransformerVar, self).__init__()

        self.memory_initial = memory_initial

        # Encoding
        self.embedding = InputEmbedding(in_dim=enc_in, d_model=d_model, dropout=dropout, device=device)
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        win_size, d_model, n_heads, dropout=dropout
                    ), d_model, d_ff, dropout=dropout, activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        self.mem_module = MemoryModule(
            n_memory=n_memory, 
            fea_dim=d_model, 
            shrink_thres=shrink_thres, 
            device=device, 
            memory_init_embedding=memory_init_embedding, 
            phase_type=phase_type, 
            dataset_name=dataset_name
        )
        
        # Decoder
        self.weak_decoder = Decoder(2 * d_model, c_out, d_ff=d_ff, activation='gelu', dropout=0.1)

    def forward(self, x):
        x = self.embedding(x)
        queries = out = self.encoder(x)
        
        outputs = self.mem_module(out)
        out, attn, memory_item_embedding = outputs['output'], outputs['attn'], outputs['memory_init_embedding']

        mem = self.mem_module.mem
        
        if self.memory_initial:
            return {"out": out, "memory_item_embedding": None, "queries": queries, "mem": mem}
        else:
            out = self.weak_decoder(out)
            return {"out": out, "memory_item_embedding": memory_item_embedding, "queries": queries, "mem": mem, "attn": attn}


# Wrapper for compatibility with existing framework
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        
        # Model parameters
        self.seq_len = args.seq_len
        self.enc_in = getattr(args, 'enc_in', 6)
        
        # MEMTO specific parameters
        self.win_size = self.seq_len
        self.c_out = self.enc_in
        self.n_memory = getattr(args, 'memto_n_memory', 100)
        self.shrink_thres = getattr(args, 'memto_shrink_thres', 0.0025)
        self.d_model = getattr(args, 'memto_d_model', 512)
        self.n_heads = getattr(args, 'memto_n_heads', 8)
        self.e_layers = getattr(args, 'memto_e_layers', 3)
        self.d_ff = getattr(args, 'memto_d_ff', 512)
        self.dropout = getattr(args, 'dropout', 0.0)
        self.activation = getattr(args, 'memto_activation', 'gelu')
        
        # Training phase and dataset
        self.phase_type = getattr(args, 'memto_phase_type', 'train')  # 'train', 'second_train', 'test'
        self.dataset_name = getattr(args, 'memto_dataset_name', 'default')
        self.memory_initial = getattr(args, 'memto_memory_initial', False)
        
        # Device
        if hasattr(args, 'gpu') and torch.cuda.is_available():
            self.device = args.gpu
        else:
            self.device = torch.device('cpu')
        
        # Initialize MEMTO model
        self.memto_model = TransformerVar(
            win_size=self.win_size,
            enc_in=self.enc_in,
            c_out=self.c_out,
            n_memory=self.n_memory,
            shrink_thres=self.shrink_thres,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            activation=self.activation,
            device=self.device,
            memory_init_embedding=None,
            memory_initial=self.memory_initial,
            phase_type=self.phase_type,
            dataset_name=self.dataset_name
        )
        
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
        outputs = self.memto_model(x)
        return outputs['queries']  # Return encoder outputs
    
    def forward(self, x, x_mark=None, y_mark=None):
        outputs = self.memto_model(x)
        
        # Store outputs for potential loss computation
        if self.training:
            self.last_outputs = outputs
            self.last_input = x
        
        return outputs['out']
    
    def compute_memto_loss(self):
        if not hasattr(self, 'last_outputs'):
            return {'total_loss': torch.tensor(0.0)}
        
        # Basic reconstruction loss
        recon_loss = F.mse_loss(self.last_outputs['out'], self.last_input)
        
        return {
            'total_loss': recon_loss,
            'recon_loss': recon_loss.item()
        }
    
    def compute_anomaly_score(self, x):
        self.eval()
        with torch.no_grad():
            outputs = self.memto_model(x)
            reconstructed = outputs['out']
            
            # Compute reconstruction error per time step
            scores = torch.mean((x - reconstructed) ** 2, dim=2)
            
            return scores
    
    def save_memory(self, save_path=None):
        if save_path is None:
            os.makedirs('./memory_item', exist_ok=True)
            save_path = f'./memory_item/{self.dataset_name}_memory_item.pth'
        
        torch.save(self.memto_model.mem_module.mem, save_path)
        print(f'Memory items saved to {save_path}')
    
    def load_memory(self, load_path=None):
        if load_path is None:
            load_path = f'./memory_item/{self.dataset_name}_memory_item.pth'
        
        if os.path.exists(load_path):
            self.memto_model.mem_module.mem = torch.load(load_path)
            print(f'Memory items loaded from {load_path}')
        else:
            print(f'Memory file {load_path} not found')