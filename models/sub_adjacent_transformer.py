
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt


# ─────────────────────────────────────────────────────────────────────────────
# Embed
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.tokenConv = nn.Conv1d(c_in, d_model, kernel_size=3,
                                   padding=1, padding_mode='circular', bias=False)
        nn.init.kaiming_normal_(self.tokenConv.weight,
                                mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super().__init__()
        self.value_embedding    = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(
            self.value_embedding(x) + self.position_embedding(x)
        )


class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None,
                 attention_dropout=0.0, output_attention=False):
        super().__init__()
        self.scale            = scale
        self.mask_flag        = mask_flag
        self.output_attention = output_attention
        self.dropout          = nn.Dropout(attention_dropout)
        tmp = torch.arange(win_size)
        self.register_buffer('distances',
                             (tmp.unsqueeze(1) - tmp.unsqueeze(0)).abs().float())

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        scale = self.scale or 1.0 / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        series = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)
        if self.output_attention:
            return V.contiguous(), series, None
        return V.contiguous(), None


class LinearAnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=False, scale=None,
                 attention_dropout=0.0, output_attention=False,
                 dim_per_head=64, mapping_fun='ours'):
        super().__init__()
        self.scale            = scale
        self.mask_flag        = mask_flag
        self.output_attention = output_attention
        self.dropout          = nn.Dropout(attention_dropout)
        self.window_size      = win_size
        self.softmax          = nn.Softmax(dim=-1)
        self.mapping_fun      = mapping_fun
        self.delta1           = nn.Parameter(torch.tensor(1.0))

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        assert L == S

        if self.mapping_fun == 'ours':
            queries = queries.clone()
            keys    = keys.clone()
            queries[queries < 0] = -100
            keys[keys < 0]       = -100
            queries = self.softmax(queries / nn.Softplus()(self.delta1))
            keys    = self.softmax(keys    / nn.Softplus()(self.delta1))
        elif self.mapping_fun == 'softmax_q_k':
            queries = self.softmax(queries)
            keys    = nn.Softmax(dim=1)(keys)
        elif self.mapping_fun == 'x_3':
            queries = nn.ReLU()(queries)
            keys    = nn.ReLU()(keys)
            q_norm  = queries.norm(dim=-1, keepdim=True)
            k_norm  = keys.norm(dim=-1, keepdim=True)
            queries = queries ** 3
            keys    = keys    ** 3
            queries = queries / (queries.norm(dim=-1, keepdim=True) + 1e-6) * q_norm.clone()
            keys    = keys    / (keys.norm(dim=-1, keepdim=True)    + 1e-6) * k_norm.clone()
        elif self.mapping_fun == 'relu':
            queries = nn.ReLU()(queries)
            keys    = nn.ReLU()(keys)
        elif self.mapping_fun == 'elu_plus_1':
            queries = F.elu(queries) + 1
            keys    = F.elu(keys)    + 1

        kv = torch.einsum("blhe,blhf->bhef", keys, values)
        z  = 1 / (torch.einsum("blhe,bhe->blh", queries,
                               keys.sum(dim=1)) + 1e-6)
        V  = torch.einsum("blhe,bhef,blh->blhf", queries, kv, z)

        if self.output_attention:
            return V.contiguous(), queries, keys
        return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None):
        super().__init__()
        d_keys   = d_keys   or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm              = nn.LayerNorm(d_model)
        self.inner_attention   = attention
        self.query_projection  = nn.Linear(d_model, d_keys   * n_heads)
        self.key_projection    = nn.Linear(d_model, d_keys   * n_heads)
        self.value_projection  = nn.Linear(d_model, d_values * n_heads)
        self.sigma_projection  = nn.Linear(d_model, n_heads)
        self.out_projection    = nn.Linear(d_values * n_heads, d_model)
        self.n_heads           = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys    = self.key_projection(keys).view(B, S, H, -1)
        values  = self.value_projection(values).view(B, S, H, -1)
        out, queries, keys = self.inner_attention(queries, keys, values)
        out = out.view(B, L, -1)
        return self.out_projection(out), queries, keys

class EncoderLayer(nn.Module):
    def __init__(self, attention_layer, d_model, d_ff=None,
                 dropout=0.1, activation='relu'):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention_layer = attention_layer
        self.conv1     = nn.Conv1d(d_model, d_ff,    kernel_size=1)
        self.conv2     = nn.Conv1d(d_ff,    d_model, kernel_size=1)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x):
        new_x, queries, keys = self.attention_layer(x, x, x)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), queries, keys


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm        = norm_layer

    def forward(self, x, attn_mask=None):
        queries_list, keys_list = [], []
        for layer in self.attn_layers:
            x, queries, keys = layer(x)
            queries_list.append(queries)
            keys_list.append(keys)
        if self.norm is not None:
            x = self.norm(x)
        return x, queries_list, keys_list

class SubAdjacentTransformerCore(nn.Module):
    def __init__(self, win_size, enc_in, c_out,
                 d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu',
                 output_attention=True, linear_attn=True,
                 mapping_fun='ours'):
        super().__init__()
        self.output_attention = output_attention
        self.embedding        = DataEmbedding(enc_in, d_model, dropout)
        dim_per_head          = d_model // n_heads

        if linear_attn:
            attn_cls = lambda: LinearAnomalyAttention(
                win_size, False,
                attention_dropout=dropout,
                output_attention=output_attention,
                dim_per_head=dim_per_head,
                mapping_fun=mapping_fun)
        else:
            attn_cls = lambda: AnomalyAttention(
                win_size, False,
                attention_dropout=dropout,
                output_attention=output_attention)

        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(attn_cls(), d_model, n_heads),
                d_model, d_ff, dropout=dropout, activation=activation
            ) for _ in range(e_layers)],
            norm_layer=nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out, queries_list, keys_list = self.encoder(enc_out)
        enc_out = self.projection(enc_out)
        if self.output_attention:
            return enc_out, queries_list, keys_list
        return enc_out

def myLossNew(queries, keys, span=None, one_side=True):
    L = queries.shape[1]
    if span is None:
        span = [20, 30]
    assert L >= span[1] >= span[0] >= 0

    attnMatrix = torch.einsum("blhe,bshe->bhls", queries, keys)
    attnMatrix = attnMatrix / (attnMatrix.sum(dim=-1, keepdim=True) + 1e-6)

    lossMat = None
    for k in range(-span[1], span[1] + 1):
        if one_side:
            if k < span[0]:
                continue
        else:
            if abs(k) < span[0]:
                continue

        diag1 = torch.diagonal(attnMatrix, offset=k, dim1=-2, dim2=-1)
        p1d   = (k, 0) if k > 0 else (0, abs(k))
        diag1 = F.pad(diag1, p1d)

        lossMat = diag1 if lossMat is None else lossMat + diag1

        offset_k = -(L - k) if k > 0 else L + k
        diag1    = torch.diagonal(attnMatrix, offset=offset_k, dim1=-2, dim2=-1)
        p1d      = (offset_k, 0) if offset_k > 0 else (0, abs(offset_k))
        diag1    = F.pad(diag1, p1d)
        lossMat  = lossMat + diag1

    return torch.mean(lossMat, dim=-2)  # B, L


def myLoss2(attnMatrix, keys=None, span=None, one_side=True):
    B, H, L, _ = attnMatrix.shape
    lossMat = None
    for k in range(20, 30):
        diag1   = torch.diagonal(attnMatrix, offset=k, dim1=-2, dim2=-1)
        diag1   = F.pad(diag1, (k, 0))
        lossMat = diag1 if lossMat is None else lossMat + diag1

        diag1   = torch.diagonal(attnMatrix, offset=-(L - k), dim1=-2, dim2=-1)
        diag1   = F.pad(diag1, (0, L - k))
        lossMat = lossMat + diag1
        lossMat = lossMat + diag1

    return torch.mean(lossMat, dim=1)  # B, L


def softmax_np(x, temperature=1, window=None):
    x = x * temperature
    shape = x.shape[0]
    if window is not None:
        window = max(min(int(window), shape), 1)
        rem = shape % window
        if rem != 0:
            x = np.concatenate([x, x[:int(window - rem)]], axis=0)
        x = x.reshape(-1, window)
    else:
        x = x.reshape(1, -1)
    x = x.clip(-100, 100)
    output = (np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)).reshape(-1)
    return output[:shape]


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class Model(nn.Module):

    def __init__(self, args):
        super().__init__()
        enc_in      = getattr(args, 'enc_in',          6)
        seq_len     = getattr(args, 'seq_len',         48)
        d_model     = getattr(args, 'd_model',        512)
        n_heads     = getattr(args, 'n_heads',          8)
        e_layers    = getattr(args, 'e_layers',         3)
        d_ff        = getattr(args, 'd_ff',           512)
        dropout     = getattr(args, 'dropout',        0.0)
        linear_attn = getattr(args, 'sat_linear_attn', True)
        mapping_fun = getattr(args, 'sat_mapping_fun', 'ours')

        self.span         = getattr(args, 'sat_span',        [20, 30])
        self.one_side     = getattr(args, 'sat_one_side',    True)
        self.k            = getattr(args, 'sat_k',           3.0)
        self.temperature  = getattr(args, 'sat_temperature', 50)
        self.softmax_span = getattr(args, 'sat_softmax_span', None)
        self.linear_attn  = linear_attn
        self.win_size     = seq_len

        assert seq_len >= self.span[1] >= self.span[0] >= 0, \
            f"seq_len({seq_len}) must be >= span[1]({self.span[1]}) >= span[0]({self.span[0]}) >= 0"

        self.core = SubAdjacentTransformerCore(
            win_size     = seq_len,
            enc_in       = enc_in,
            c_out        = enc_in,
            d_model      = d_model,
            n_heads      = n_heads,
            e_layers     = e_layers,
            d_ff         = d_ff,
            dropout      = dropout,
            activation   = 'gelu',
            output_attention = True,
            linear_attn  = linear_attn,
            mapping_fun  = mapping_fun,
        )
        self._loss_fn = myLossNew if linear_attn else myLoss2

    def forward(self, x, x_mark=None, y_mark=None):
        output, queries_list, keys_list = self.core(x)
        self._last_queries = queries_list
        self._last_keys    = keys_list
        return output

    def compute_loss(self, x, output, phase=2, rec_criterion=None):
        if rec_criterion is None:
            criterion = nn.MSELoss()
            rec_loss  = criterion(output, x)
        else:
            rec_loss  = rec_criterion(output, x)

        queries_list = self._last_queries
        keys_list    = self._last_keys
        n = len(queries_list)

        loss_attn = sum(
            self._loss_fn(queries_list[u], keys_list[u],
                          self.span, self.one_side).mean()
            for u in range(n)
        ) / n

        if phase == 1:
            return rec_loss
        else:
            return 2 * rec_loss - self.k * loss_attn

    @torch.no_grad()
    def anomaly_score_batch(self, x):
        criterion = nn.MSELoss(reduction='none')
        output, queries_list, keys_list = self.core(x)
        n = len(queries_list)

        # rec_loss: [B, L]
        rec_loss = torch.mean(criterion(x, output), dim=-1)

        loss_attn = sum(
            self._loss_fn(queries_list[u], keys_list[u],
                          self.span, self.one_side)
            for u in range(n)
        ) / n  # [B, L]

        loss_attn_np = loss_attn.detach().cpu().numpy()  # [B, L]
        rec_loss_np  = rec_loss.detach().cpu().numpy()   # [B, L]

        scores = []
        for b in range(x.size(0)):
            attn_b = loss_attn_np[b]   # [L]
            rec_b  = rec_loss_np[b]    # [L]
            sm     = softmax_np(-attn_b, temperature=self.temperature,
                                window=self.softmax_span)
            scores.append(sm * rec_b)  # [L]

        return np.stack(scores, axis=0)  # [B, L]