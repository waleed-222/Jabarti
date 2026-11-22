import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_k = config.d_k

        # One full-width projection per role, carved into heads in forward().
        self.W_q = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)
        self.W_o = nn.Linear(config.d_model, config.d_model)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, kv_cache=None, use_cache=False):
        batch_size, seq_len, d_model = x.shape

        # (B, T, d_model) -> (B, n_heads, T, d_k)
        q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        past_len = 0
        if kv_cache is not None:
            past_key, past_value = kv_cache
            past_len = past_key.size(2)
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)

        updated_cache = (k, v) if use_cache else None
        total_len = past_len + seq_len

        scores = (q @ k.transpose(-2, -1)) / self.d_k**0.5          # (B, n_heads, T, total_len)
        causal = self.mask[past_len:total_len, :total_len]          # (T, total_len)
        scores = scores.masked_fill(~causal, float("-inf"))

        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        out = weights @ v                                           # (B, n_heads, T, d_k)

        # (B, n_heads, T, d_k) -> (B, T, n_heads, d_k) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.resid_dropout(self.W_o(out))

        return out, updated_cache