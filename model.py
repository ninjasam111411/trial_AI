import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import EOS


class NeuralChatbot(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size=192,
        heads=6,
        layers=4,
        context_size=256,
        dropout=0.1
    ):
        super().__init__()

        self.context_size = context_size

        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(context_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_size, heads, dropout)
            for _ in range(layers)
        ])

        self.norm = nn.LayerNorm(embedding_size)
        self.output = nn.Linear(embedding_size, vocab_size, bias=False)

        # Share weights between input and output embeddings:
        # fewer parameters and usually better results on small data.
        self.output.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        batch, length = x.shape
        positions = torch.arange(length, device=x.device)

        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        mask = torch.triu(
            torch.ones(length, length, device=x.device, dtype=torch.bool),
            diagonal=1
        )

        for block in self.blocks:
            x = block(x, mask)

        return self.output(self.norm(x))

    @torch.no_grad()
    def generate(
        self,
        x,
        max_tokens=300,
        temperature=0.7,
        top_k=20,
        stop_text_ids=None
    ):
        """
        Generate tokens until the end-of-response token, or until the
        sequence in stop_text_ids (e.g. "\\nUser:") appears.
        """
        start = x.size(1)

        for _ in range(max_tokens):
            x_context = x[:, -self.context_size:]
            logits = self(x_context)[:, -1, :] / max(temperature, 1e-5)

            if top_k:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")

            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

            if next_token.item() == EOS:
                break

            x = torch.cat((x, next_token), dim=1)

            if stop_text_ids:
                n = len(stop_text_ids)
                tail = x[0, -n:].tolist()
                if x.size(1) - start >= n and tail == stop_text_ids:
                    x = x[:, :-n]
                    break

        return x


class TransformerBlock(nn.Module):
    """Pre-norm transformer block (trains more stably than post-norm)."""

    def __init__(self, embedding_size, heads, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_size)
        self.attention = nn.MultiheadAttention(
            embedding_size,
            heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embedding_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 4),
            nn.GELU(),
            nn.Linear(embedding_size * 4, embedding_size),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.norm1(x)
        attention, _ = self.attention(
            h, h, h, attn_mask=mask, need_weights=False
        )
        x = x + self.dropout(attention)
        x = x + self.feed_forward(self.norm2(x))
        return x
