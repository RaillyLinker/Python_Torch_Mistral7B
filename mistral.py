import math
import torch
from torch import nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.scale = math.sqrt(emb_size)

    def forward(self, tokens):
        return self.embedding(tokens.long()) * self.scale


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, dropout=0.1):
        super().__init__()
        self.token = TokenEmbedding(vocab_size, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        token_emb = self.token(x)
        return self.dropout(token_emb)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def build_rotary_pos_emb(dim, max_seq_len=2048):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.einsum('i , j -> i j', t, inv_freq)  # (max_seq_len, dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (max_seq_len, dim)
    cos = emb.cos()[None, None, :, :]  # (1, 1, max_seq_len, dim)
    sin = emb.sin()[None, None, :, :]
    return cos, sin


def apply_rotary_pos_emb_single(q, k, cos, sin, seq_len):
    cos = cos[:, :, :seq_len, :]
    sin = sin[:, :, :seq_len, :]
    q_ = (q * cos) + (rotate_half(q) * sin)
    k_ = (k * cos) + (rotate_half(k) * sin)
    return q_, k_


class GqaAndSwa(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, window_size, dropout=0.0, max_seq_len=2048):
        super().__init__()
        assert d_model % num_heads == 0
        assert d_model % num_kv_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.kv_head_dim = d_model // num_kv_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.window_size = window_size

        cos, sin = build_rotary_pos_emb(self.head_dim, max_seq_len)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)

    def _make_sliding_window_mask(self, seq_len, device):
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        for i in range(seq_len):
            start = max(0, i - self.window_size + 1)
            mask[i, start:i + 1] = 0
        return mask

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)

        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.kv_head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.kv_head_dim)
        group_size = self.num_heads // self.num_kv_heads

        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        k = k.view(batch_size, self.num_kv_heads, seq_len, group_size, self.head_dim)
        v = v.view(batch_size, self.num_kv_heads, seq_len, group_size, self.head_dim)

        k = k.permute(0, 1, 3, 2, 4).contiguous().view(batch_size, self.num_heads, seq_len, self.head_dim)
        v = v.permute(0, 1, 3, 2, 4).contiguous().view(batch_size, self.num_heads, seq_len, self.head_dim)

        q, k = apply_rotary_pos_emb_single(q, k, self.cos, self.sin, seq_len)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        sliding_mask = self._make_sliding_window_mask(seq_len, x.device)
        sliding_mask = sliding_mask.unsqueeze(0).unsqueeze(0)

        if mask is not None:
            combined_mask = torch.maximum(mask.float(), sliding_mask)
        else:
            combined_mask = sliding_mask

        scores = scores + combined_mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = RMSNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * F.silu(x2)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff * 2, bias=False)
        self.activation = SwiGLU()
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return self.dropout(x)


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, hidden_dim, window_size=4096, dropout=0.1):
        super().__init__()
        self.attn = GqaAndSwa(dim, num_heads, num_kv_heads, window_size, dropout)
        self.ff = PositionwiseFeedForward(dim, hidden_dim, dropout)
        self.sublayers = nn.ModuleList([
            SublayerConnection(dim, 0.0),
            SublayerConnection(dim, 0.0)
        ])

    def forward(self, x):
        x = self.sublayers[0](x, lambda _x: self.attn(_x))
        x = self.sublayers[1](x, self.ff)
        return x


class Mistral7B(nn.Module):
    def __init__(
            self,
            vocab_size: int = 32000,
            dim: int = 4096,
            n_layers: int = 32,
            num_heads: int = 32,
            num_kv_heads: int = 8,
            hidden_dim: int = 14336,
            max_len: int = 131072,
            window_size: int = 4096,
            dropout=0.1
    ):
        super().__init__()
        self.max_len = max_len
        self.embed = InputEmbedding(vocab_size, dim, dropout)
        self.layers = nn.ModuleList([
            DecoderBlock(dim, num_heads, num_kv_heads, hidden_dim, window_size, dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.embed.token.embedding.weight

    def forward(self, x):
        batch_size, seq_len = x.size()
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max length {self.max_len}")

        x = self.embed(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.head(x)
        return logits
