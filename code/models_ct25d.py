"""DINOv3 2.5D volume regressors (Abdominal Age from CT, Brain Age from T1w MRI)."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

from models import MODEL_REGISTRY, PredictionHead


AGGREGATORS = {
    "axial_mean",
    "axial_transformer",
    "triplanar_hierarchical",
}


class MaskedAttentionPool(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.score = nn.Linear(embed_dim, 1)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        scores = self.score(tokens).squeeze(-1)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        empty = ~valid.any(dim=1)
        if empty.any():
            scores = scores.clone()
            scores[empty] = 0
        weights = torch.softmax(scores, dim=1) * valid.to(scores.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.sum(tokens * weights.unsqueeze(-1), dim=1)


def _safe_transformer(
    encoder: nn.Module,
    tokens: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Avoid an all-padding Transformer row while preserving its zero mask."""
    safe_valid = valid.clone()
    empty = ~safe_valid.any(dim=1)
    if empty.any():
        safe_valid[empty, 0] = True
    return encoder(tokens, src_key_padding_mask=~safe_valid)


class CT25DBiomarkerModel(nn.Module):
    """Shared 2D DINO slice encoder plus an axial or tri-planar aggregator."""

    def __init__(
        self,
        version: str = "dinov3",
        model: str = "large",
        pretrained: bool = True,
        img_size: int = 224,
        grad_checkpointing: bool = False,
        head_hidden_dim: int = 32,
        head_dropout: float = 0.5,
        aggregator: str = "triplanar_hierarchical",
        encoder_chunk_size: int = 4,
        aggregator_dim: int = 256,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        transformer_dropout: float = 0.1,
    ):
        super().__init__()
        if aggregator not in AGGREGATORS:
            raise ValueError(f"Unknown CT aggregator {aggregator}; choose {AGGREGATORS}")
        if encoder_chunk_size <= 0:
            raise ValueError("encoder_chunk_size must be positive")
        if aggregator_dim % transformer_heads:
            raise ValueError("aggregator_dim must be divisible by transformer_heads")

        self.aggregator_name = aggregator
        self.encoder_chunk_size = int(encoder_chunk_size)
        self.grad_checkpointing = bool(grad_checkpointing)
        model_name = MODEL_REGISTRY[(version, model)]
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
        )
        self.slice_embed_dim = int(self.backbone.embed_dim)

        if aggregator == "axial_mean":
            self.head = PredictionHead(
                embed_dim=self.slice_embed_dim,
                hidden_dim=head_hidden_dim,
                dropout=head_dropout,
            )
            return

        self.input_projection = nn.Linear(self.slice_embed_dim, aggregator_dim)
        self.coordinate_embedding = nn.Sequential(
            nn.Linear(1, aggregator_dim),
            nn.GELU(),
            nn.Linear(aggregator_dim, aggregator_dim),
        )
        self.view_embedding = nn.Embedding(3, aggregator_dim)

        def make_encoder() -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=aggregator_dim,
                nhead=transformer_heads,
                dim_feedforward=aggregator_dim * 4,
                dropout=transformer_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=transformer_layers)

        self.slice_pool = MaskedAttentionPool(aggregator_dim)
        if aggregator == "axial_transformer":
            self.axial_encoder = make_encoder()
        else:
            self.view_encoders = nn.ModuleList([make_encoder() for _ in range(3)])
            self.fusion_view_embedding = nn.Embedding(3, aggregator_dim)
            self.cross_view_encoder = make_encoder()
            self.cross_view_pool = MaskedAttentionPool(aggregator_dim)
        self.aggregator_norm = nn.LayerNorm(aggregator_dim)
        self.head = PredictionHead(
            embed_dim=aggregator_dim,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        )

    def set_training_mode(self, mode: str, k: int = 0) -> None:
        if mode not in {"linear_probe", "finetune_last_k", "finetune_all"}:
            raise ValueError(f"Unknown train_mode: {mode}")

        # Make mode changes deterministic even if this method is called more
        # than once on the same model instance.
        for parameter in self.backbone.parameters():
            parameter.requires_grad = mode == "finetune_all"
        if mode == "finetune_last_k":
            if k <= 0:
                raise ValueError("finetune_last_k requires finetune_k > 0")
            for block in self.backbone.blocks[-k:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            if hasattr(self.backbone, "fc_norm"):
                for parameter in self.backbone.fc_norm.parameters():
                    parameter.requires_grad = True

        enable_checkpointing = self.grad_checkpointing and mode != "linear_probe"
        checkpoint_setter = getattr(self.backbone, "set_grad_checkpointing", None)
        if enable_checkpointing and checkpoint_setter is None:
            raise RuntimeError(
                "Backbone does not support the requested gradient checkpointing"
            )
        if checkpoint_setter is not None:
            checkpoint_setter(enable_checkpointing)

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def no_weight_decay(self) -> set[str]:
        result = {
            "view_embedding.weight",
            "fusion_view_embedding.weight",
        }
        if hasattr(self.backbone, "no_weight_decay"):
            result.update(
                f"backbone.{name}" for name in self.backbone.no_weight_decay()
            )
        return result

    def _pool_patch_tokens(self, features) -> torch.Tensor:
        if isinstance(features, dict):
            if "x_norm_patchtokens" in features:
                pooled = features["x_norm_patchtokens"].mean(dim=1)
                return self.backbone.fc_norm(pooled)
            for key in ("x", "last_hidden_state"):
                if key in features:
                    features = features[key]
                    break
            else:
                raise ValueError(
                    f"Unsupported backbone feature dictionary: {features.keys()}"
                )
        if features.ndim == 2:
            return features
        if features.ndim != 3:
            raise ValueError(f"Expected BxTxD backbone features, got {features.shape}")
        prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))
        pooled = features[:, prefix_tokens:, :].mean(dim=1)
        return self.backbone.fc_norm(pooled)

    def _encode_slice_chunk(self, images: torch.Tensor) -> torch.Tensor:
        return self._pool_patch_tokens(self.backbone.forward_features(images))

    def _encode_valid_slices(
        self,
        images: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, slice_count = images.shape[:2]
        flat_images = images.flatten(0, 1)
        flat_valid = valid.flatten()
        valid_indices = torch.nonzero(flat_valid, as_tuple=False).squeeze(1)
        if not len(valid_indices):
            raise ValueError("The batch contains no valid CT slices")

        backbone_trainable = any(
            parameter.requires_grad for parameter in self.backbone.parameters()
        )
        encoded_chunks = []
        for start in range(0, len(valid_indices), self.encoder_chunk_size):
            indices = valid_indices[start : start + self.encoder_chunk_size]
            chunk = flat_images.index_select(0, indices)
            if backbone_trainable:
                encoded = self._encode_slice_chunk(chunk)
            else:
                with torch.no_grad():
                    encoded = self._encode_slice_chunk(chunk)
            encoded_chunks.append(encoded)
        encoded_valid = torch.cat(encoded_chunks, dim=0)
        flat_features = encoded_valid.new_zeros(
            (batch_size * slice_count, self.slice_embed_dim)
        )
        flat_features = flat_features.index_copy(0, valid_indices, encoded_valid)
        return flat_features.view(batch_size, slice_count, self.slice_embed_dim)

    @staticmethod
    def _check_samples(samples: dict[str, torch.Tensor]) -> None:
        required = {"images", "valid", "coordinates", "view_ids"}
        if not required.issubset(samples):
            raise ValueError(f"Missing CT sample fields: {required - set(samples)}")
        images = samples["images"]
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(f"Expected BxSx3xHxW CT images, got {images.shape}")
        expected = images.shape[:2]
        for key in ("valid", "coordinates", "view_ids"):
            if samples[key].shape != expected:
                raise ValueError(f"{key} shape {samples[key].shape} != {expected}")

    def forward(self, samples: dict[str, torch.Tensor]) -> torch.Tensor:
        self._check_samples(samples)
        images = samples["images"]
        valid = samples["valid"].bool()
        coordinates = samples["coordinates"].to(images.dtype)
        view_ids = samples["view_ids"].long()
        axial_valid = valid & view_ids.eq(0)
        if not axial_valid.any(dim=1).all():
            raise ValueError("Each CT volume must contain at least one valid axial slice")

        encode_valid = axial_valid if self.aggregator_name != "triplanar_hierarchical" else valid
        slice_features = self._encode_valid_slices(images, encode_valid)

        if self.aggregator_name == "axial_mean":
            weights = axial_valid.to(slice_features.dtype).unsqueeze(-1)
            pooled = (slice_features * weights).sum(dim=1)
            pooled = pooled / weights.sum(dim=1).clamp_min(1.0)
            return self.head(pooled)

        tokens = self.input_projection(slice_features)
        tokens = tokens + self.coordinate_embedding(coordinates.unsqueeze(-1))
        tokens = tokens + self.view_embedding(view_ids)

        if self.aggregator_name == "axial_transformer":
            encoded = _safe_transformer(self.axial_encoder, tokens, axial_valid)
            pooled = self.slice_pool(encoded, axial_valid)
        else:
            view_tokens = []
            view_valid = []
            for view_index, encoder in enumerate(self.view_encoders):
                mask = valid & view_ids.eq(view_index)
                encoded = _safe_transformer(encoder, tokens, mask)
                view_tokens.append(self.slice_pool(encoded, mask))
                view_valid.append(mask.any(dim=1))
            view_tokens = torch.stack(view_tokens, dim=1)
            view_valid = torch.stack(view_valid, dim=1)
            fusion_ids = torch.arange(3, device=view_tokens.device).view(1, 3)
            view_tokens = view_tokens + self.fusion_view_embedding(fusion_ids)
            fused = _safe_transformer(
                self.cross_view_encoder,
                view_tokens,
                view_valid,
            )
            pooled = self.cross_view_pool(fused, view_valid)
        return self.head(self.aggregator_norm(pooled))
