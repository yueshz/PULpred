import torch
import torch.nn as nn


class PULTransformer(nn.Module):
    """
    Set Transformer for PUL pre-training and fine-tuning.

    No positional encoding — gene order is treated as irrelevant.

    Architecture:
        input_proj  : Linear(input_dim, d_model)
        [CLS] token : learnable, prepended to every PUL
        [MASK] token: learnable, replaces masked proteins during pre-training
        encoder     : N-layer TransformerEncoder (Pre-LN, batch_first)
        pretrain_head : MLP(d_model → input_dim)  used during pre-training
        finetune_head : Linear(d_model → num_classes)  attached for fine-tuning

    Args:
        input_dim      : dimension of ESM2 protein embeddings (640)
        d_model        : internal Transformer dimension (512)
        nhead          : number of attention heads (8)
        num_layers     : number of Transformer encoder layers (4)
        dim_feedforward: FFN hidden dimension (2048)
        dropout        : dropout rate (0.1)
        num_classes    : number of substrate classes; set when fine-tuning
    """

    def __init__(
        self,
        input_dim: int = 640,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_classes: int | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model   = d_model

        self.input_proj = nn.Linear(input_dim, d_model)

        # Special tokens (no positional encoding — order invariant)
        self.cls_token  = nn.Parameter(torch.empty(1, 1, d_model))
        self.mask_token = nn.Parameter(torch.empty(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # Pre-training head: reconstruct masked protein embeddings
        self.pretrain_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, input_dim),
        )

        # Fine-tuning head (attached later via attach_finetune_head)
        self.finetune_head: nn.Linear | None = None
        if num_classes is not None:
            self.attach_finetune_head(num_classes)

        self._init_weights()

    # ── Initialisation ─────────────────────────────────────────────────────────

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token,  std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def attach_finetune_head(self, num_classes: int):
        self.finetune_head = nn.Linear(self.d_model, num_classes)
        nn.init.trunc_normal_(self.finetune_head.weight, std=0.02)
        nn.init.zeros_(self.finetune_head.bias)

    # ── Core encoder ───────────────────────────────────────────────────────────

    def encode(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        masked_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            embeddings      : [B, L, input_dim]
            attention_mask  : [B, L] bool  — True = real protein, False = padding
            masked_positions: [B, L] bool  — True = replace with mask token

        Returns:
            x : [B, L+1, d_model]  (position 0 is CLS)
        """
        B, L, _ = embeddings.shape

        x = self.input_proj(embeddings)   # [B, L, d_model]

        # Replace masked positions with the learned mask token
        if masked_positions is not None:
            mask_exp = masked_positions.unsqueeze(-1).expand_as(x)
            x = torch.where(mask_exp, self.mask_token.expand(B, L, -1), x)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)      # [B, 1, d_model]
        x   = torch.cat([cls, x], dim=1)             # [B, L+1, d_model]

        # Build full padding mask (CLS is never masked)
        cls_valid   = torch.ones(B, 1, dtype=torch.bool, device=x.device)
        full_mask   = torch.cat([cls_valid, attention_mask], dim=1)  # [B, L+1]
        padding_mask = ~full_mask   # TransformerEncoder: True = ignore this position

        return self.encoder(x, src_key_padding_mask=padding_mask)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        masked_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Returns:
            cls_repr : [B, d_model]          — PUL-level representation
            pred_emb : [B, L, input_dim] | None — reconstructed protein embeddings
            logits   : [B, num_classes] | None  — substrate class logits
        """
        x = self.encode(embeddings, attention_mask, masked_positions)

        cls_repr = x[:, 0, :]      # [B, d_model]
        tokens   = x[:, 1:, :]    # [B, L, d_model]

        pred_emb = self.pretrain_head(tokens) if masked_positions is not None else None
        logits   = self.finetune_head(cls_repr) if self.finetune_head is not None else None

        return cls_repr, pred_emb, logits

    # ── Convenience ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_pul_embedding(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return CLS-token representation without gradient."""
        cls_repr, _, _ = self.forward(embeddings, attention_mask)
        return cls_repr

    def freeze_encoder(self):
        """Freeze everything except the fine-tuning head."""
        for name, p in self.named_parameters():
            if "finetune_head" not in name:
                p.requires_grad_(False)

    def unfreeze_encoder(self):
        for p in self.parameters():
            p.requires_grad_(True)
