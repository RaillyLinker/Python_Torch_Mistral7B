import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.scale = math.sqrt(emb_size)

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        # tokens: (batch, seq)
        # output: (batch, seq, emb_size) scaled by sqrt(emb_size)
        return self.embedding(tokens) * self.scale


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.token = TokenEmbedding(vocab_size, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        # x: (batch, seq)
        tok_emb = self.token(x)  # (batch, seq, d_model)
        return self.dropout(tok_emb)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # (.., 2*k) -> (.., k), (.., k)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def build_rotary_pos_emb(dim: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (seq_len, dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
    cos = emb.cos()[None, None, :, :]  # (1, 1, seq_len, dim)
    sin = emb.sin()[None, None, :, :]  # (1, 1, seq_len, dim)
    return cos, sin


def apply_rotary_pos_emb_single(q, k, cos, sin, seq_len, dtype):
    cos = cos[:, :, :seq_len, :].to(dtype=dtype)
    sin = sin[:, :, :seq_len, :].to(dtype=dtype)
    q_ = (q * cos) + (rotate_half(q) * sin)
    k_ = (k * cos) + (rotate_half(k) * sin)
    return q_, k_


class GqaAndSwa(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            num_kv_heads: int,
            window_size: int,
            dropout: float = 0.1,
            max_seq_len: int = 131072,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert d_model % num_kv_heads == 0, "d_model must be divisible by num_kv_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.kv_head_dim = d_model // num_kv_heads
        self.window_size = window_size
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        cos_buf, sin_buf = build_rotary_pos_emb(self.head_dim, max_seq_len)
        self.register_buffer("rotary_cos", cos_buf, persistent=True)
        self.register_buffer("rotary_sin", sin_buf, persistent=True)

        # Cache for sliding window masks, keyed by (seq_len, device)
        self._mask_cache = {}

    def _make_sliding_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        cache_key = (seq_len, device)
        if cache_key not in self._mask_cache:
            i = torch.arange(seq_len, device=device).unsqueeze(1)
            j = torch.arange(seq_len, device=device).unsqueeze(0)
            diff = i - j  # (seq_len, seq_len)
            base_mask = torch.where(
                (diff < 0) | (diff >= self.window_size),
                float("-inf"),
                0.0
            )
            self._mask_cache[cache_key] = base_mask
        return self._mask_cache[cache_key]

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            global_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        device = x.device
        dtype = x.dtype

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # (batch, num_heads, seq, head_dim)

        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.kv_head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.kv_head_dim)

        group_size = self.num_heads // self.num_kv_heads

        k = k.permute(0, 2, 1, 3)  # (batch, num_kv_heads, seq, kv_head_dim)
        v = v.permute(0, 2, 1, 3)

        k = k.view(batch_size, self.num_kv_heads, seq_len, group_size, self.head_dim)
        v = v.view(batch_size, self.num_kv_heads, seq_len, group_size, self.head_dim)

        k = k.permute(0, 1, 3, 2, 4).contiguous().view(batch_size, self.num_heads, seq_len, self.head_dim)
        v = v.permute(0, 1, 3, 2, 4).contiguous().view(batch_size, self.num_heads, seq_len, self.head_dim)

        q, k = apply_rotary_pos_emb_single(q, k, self.rotary_cos, self.rotary_sin, seq_len, dtype)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch, num_heads, seq, seq)

        sliding_mask = self._make_sliding_window_mask(seq_len, device)  # (seq, seq)
        sliding_mask = sliding_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)

        if mask is not None:
            combined_mask = mask + sliding_mask
        else:
            combined_mask = sliding_mask

        if global_token_mask is not None:
            gt = global_token_mask.unsqueeze(1).unsqueeze(3)  # (batch, 1, seq, 1)
            gs = global_token_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq)
            combined_mask = torch.where(
                (gt | gs),
                torch.zeros_like(combined_mask),
                combined_mask
            )

        scores = scores + combined_mask  # Add -inf where we need to mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (batch, num_heads, seq, head_dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, dim)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class SublayerConnection(nn.Module):
    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.norm = RMSNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: callable) -> torch.Tensor:
        # Pre-norm → sublayer → dropout → residual add
        return x + self.dropout(sublayer(self.norm(x)))


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * F.silu(x2)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff * 2, bias=False)
        self.activation = SwiGLU()
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)  # (batch, seq, d_ff*2)
        x = self.activation(x)  # (batch, seq, d_ff)
        x = self.linear2(x)  # (batch, seq, d_model)
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
