
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralChatbot(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size=128,
        heads=4,
        layers=2,
        context_size=128
    ):
        super().__init__()

        self.context_size = context_size

        self.token_embedding = nn.Embedding(
            vocab_size, embedding_size
        )

        self.position_embedding = nn.Embedding(
            context_size, embedding_size
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_size, heads)
            for _ in range(layers)
        ])

        self.norm = nn.LayerNorm(embedding_size)
        self.output = nn.Linear(embedding_size, vocab_size)

    def forward(self, x):
        batch, length = x.shape

        positions = torch.arange(
            length, device=x.device
        )

        x = (
            self.token_embedding(x)
            + self.position_embedding(positions)
        )

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return self.output(x)

    @torch.no_grad()
    def generate(self, x, max_tokens=300, temperature=0.8):
        for _ in range(max_tokens):

            x_context = x[:, -self.context_size:]

            logits = self(x_context)

            logits = logits[:, -1, :] / temperature

            probabilities = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(
                probabilities, num_samples=1
            )

            x = torch.cat((x, next_token), dim=1)

            if next_token.item() == 0:
                break

        return x


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, heads):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embedding_size,
            heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 4),
            nn.ReLU(),
            nn.Linear(embedding_size * 4, embedding_size)
        )

    def forward(self, x):
        length = x.size(1)

        mask = torch.triu(
            torch.ones(
                length, length,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        attention, _ = self.attention(
            x, x, x, attn_mask=mask,
            need_weights=False
        )

        x = self.norm1(x + attention)

        x = self.norm2(x + self.feed_forward(x))

        return x
