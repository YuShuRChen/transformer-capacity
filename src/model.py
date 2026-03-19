import torch
import torch.nn as nn


class BasicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, ffn_multiplier, n_layers, max_len=256, init_seed=None):
        # seed for model initialization
        # if init_seed is not None:
        #     prev_rng_state = torch.get_rng_state()
        #     torch.manual_seed(init_seed)

        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ffn_multiplier,
            dropout=0.0,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=n_layers, enable_nested_tensor=False)

        self.lm_head = nn.Linear(d_model, vocab_size)

        mask = nn.Transformer.generate_square_subsequent_mask(max_len)
        self.register_buffer("mask", mask)

        # if init_seed is not None:
        #     torch.set_rng_state(prev_rng_state)

    def forward(self, x):
        b, t = x.size()

        x = self.embedding(x) + self.pos_embedding[:, :t, :]

        mask = self.mask[:t, :t]
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, prefix, total_length):
        self.eval()
        device = next(self.parameters()).device

        max_len = self.pos_embedding.size(1)
        assert total_length <= max_len, f"Total length {total_length} exceeds max length {max_len}"

        is_1d = False
        if prefix.dim() == 1:
            is_1d = True
            prefix = prefix.unsqueeze(0)
        generated = prefix.to(device)

        for _ in range(total_length - prefix.size(1)):
            logits = self(generated)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

        if is_1d:
            generated = generated.squeeze(0)
        return generated
