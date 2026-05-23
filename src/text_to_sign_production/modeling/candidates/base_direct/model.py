"""Direct M0 model construction backed by the existing baseline architecture."""

from __future__ import annotations

from text_to_sign_production.modeling.candidates.base_direct.config import BaseDirectRunConfig


def build_base_direct_model(config: BaseDirectRunConfig):
    """Build the direct transcript-to-continuous-BFH baseline model."""

    from text_to_sign_production.modeling.backbones.flan_t5 import FlanT5TextBackbone
    from text_to_sign_production.modeling.models.baseline import BaselineTextToPoseModel

    backbone = FlanT5TextBackbone(
        model_name=config.text_encoder.model_name,
        revision=config.text_encoder.revision,
        max_length=config.text_encoder.max_length,
        local_files_only=config.text_encoder.local_files_only,
        trainable=config.text_encoder.trainable,
        freeze_strategy=config.text_encoder.freeze_strategy,
    )
    return BaselineTextToPoseModel(
        backbone,
        decoder_hidden_dim=config.model.decoder_hidden_dim,
        decoder_layers=config.model.decoder_layers,
        decoder_dropout=config.model.decoder_dropout,
        frame_position_encoding_dim=config.model.frame_position_encoding_dim,
    )
__all__ = ["build_base_direct_model"]
