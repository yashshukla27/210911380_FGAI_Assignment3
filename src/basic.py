"""A randomly initialized, genuine encoder-decoder Transformer."""
import math
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

PAD, UNK, BOS, EOS = 0, 1, 2, 3


class BasicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, layers=3, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.scale = math.sqrt(d_model)
        pe = torch.zeros(2048, d_model)
        pos = torch.arange(2048).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos*div), torch.cos(pos*div)
        self.register_buffer('positions', pe.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
            num_encoder_layers=layers, num_decoder_layers=layers,
            dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.output = nn.Linear(d_model, vocab_size)

    def embed(self, x):
        if x.shape[1] > self.positions.shape[1]:
            raise ValueError('Sequence exceeds positional encoding capacity')
        return self.dropout(self.embedding(x)*self.scale + self.positions[:, :x.shape[1]])

    def forward(self, src, tgt):
        mask = torch.triu(torch.ones(tgt.shape[1], tgt.shape[1], device=tgt.device, dtype=torch.bool), 1)
        h = self.transformer(self.embed(src), self.embed(tgt), tgt_mask=mask,
            src_key_padding_mask=src.eq(PAD), tgt_key_padding_mask=tgt.eq(PAD),
            memory_key_padding_mask=src.eq(PAD))
        return self.output(h)

    @torch.inference_mode()
    def translate(self, source, sp, device, max_new_tokens=128, temperature=None):
        self.eval()
        src = torch.tensor([sp.encode(source)+[EOS]], device=device)
        src_pad = src.eq(PAD)
        memory = self.transformer.encoder(self.embed(src), src_key_padding_mask=src_pad)
        out = torch.tensor([[BOS]], device=device)
        for _ in range(max_new_tokens):
            mask = torch.triu(torch.ones(out.shape[1], out.shape[1], device=device, dtype=torch.bool), 1)
            h = self.transformer.decoder(self.embed(out), memory, tgt_mask=mask, memory_key_padding_mask=src_pad)
            logits = self.output(h[:, -1]).float()
            logits[:, PAD] = logits[:, BOS] = -float('inf')
            nxt = logits.argmax(-1, keepdim=True) if temperature is None else torch.multinomial(
                torch.softmax(logits/temperature, dim=-1), 1)
            out = torch.cat([out, nxt], dim=1)
            if nxt.item() == EOS:
                break
        tokens = out[0, 1:].tolist()
        return sp.decode([t for t in tokens if t != EOS]), tokens[-1] != EOS


def collate_basic(rows, sp):
    src = [torch.tensor(sp.encode(r['en'])+[EOS]) for r in rows]
    tgt = [torch.tensor([BOS]+sp.encode(r['hi'])+[EOS]) for r in rows]
    return pad_sequence(src, batch_first=True, padding_value=PAD), pad_sequence(tgt, batch_first=True, padding_value=PAD)
