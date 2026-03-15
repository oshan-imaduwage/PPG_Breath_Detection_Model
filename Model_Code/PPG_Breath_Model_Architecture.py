"""
Module 4: Neural Network Architecture
======================================
Parallel multi-branch 1D-CNN with dual outputs (classifier + regressor).
Three independent encoders feed a fusion core and a primary-only regularisation
path. Both paths produce classification logits; only the fusion path produces
the final regression location.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import sys
sys.path.append("/content/drive/MyDrive/Colab_Notebooks")
from PPG_Breath_Model_Preprocessing import Config, BIDMCLoader
from PPG_Breath_Model_Dataset import RespiratoryWindowDataset, make_loader


# ─────────────────────────────────────────────────────────────────────────────
# BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv1D → BatchNorm1D → ReLU → (optional) MaxPool1D."""

    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 7, pool: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel,
                      padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    """
    Lightweight single-channel 1D-CNN encoder.

    Produces a temporal feature map that is later concatenated with other
    encoders before being passed to the core network.

    Input  : (B, 1, T)
    Output : (B, channels[-1], T // 2^N)
    """

    def __init__(self, channels: list[int], kernel: int = 7):
        super().__init__()
        self.blocks = nn.ModuleList()
        in_ch = 1
        for out_ch in channels:
            self.blocks.append(ConvBlock(in_ch, out_ch, kernel, pool=True))
            in_ch = out_ch
        self.out_channels = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class CoreNetwork(nn.Module):
    """
    Deeper 1D-CNN + fully connected network.

    Takes the (possibly concatenated) encoder outputs and produces a dense
    feature vector of size `dense_units`.

    Input  : (B, in_ch, T')
    Output : (B, dense_units)

    The dense layer size is determined lazily on the first forward pass,
    so no manual calculation of the post-conv temporal dimension is needed.
    """

    def __init__(self, in_ch: int, conv_channels: list[int],
                 dense_units: int, dropout: float):
        super().__init__()
        self.dense_units = dense_units
        self.dropout_p   = dropout

        # Convolutional layers
        conv_layers: list[nn.Module] = []
        ch = in_ch
        for i, out_ch in enumerate(conv_channels):
            use_pool = (i < len(conv_channels) - 1)     # no pool on last layer
            conv_layers.append(ConvBlock(ch, out_ch, kernel=5, pool=use_pool))
            ch = out_ch
        conv_layers.append(nn.Flatten())
        self.convs = nn.Sequential(*conv_layers)

        # Dense layers (built lazily)
        self._dense: nn.Sequential | None = None

    def _build_dense(self, flat_dim: int, device: torch.device):
        D = self.dense_units
        self._dense = nn.Sequential(
            nn.Linear(flat_dim, D * 2),
            nn.BatchNorm1d(D * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p),
            nn.Linear(D * 2, D),
            nn.BatchNorm1d(D),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p),
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        if self._dense is None:
            self._build_dense(x.shape[1], x.device)
        return self._dense(x)




class ClassificationHead(nn.Module):
    """
    Two-layer MLP producing un-normalised logits (NOT passed through sigmoid).
    BCEWithLogitsLoss is used during training for numerical stability.

    Output : (B, num_transitions)   ← raw logits
    """

    def __init__(self, in_features: int, num_transitions: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.BatchNorm1d(in_features // 2),
            nn.Dropout(dropout),
            nn.Softmax(dim=1),
            nn.Linear(in_features // 2, num_transitions),
            # No activation — logits passed directly to BCEWithLogitsLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegressionHead(nn.Module):
    """
    Two-layer MLP producing a sample-index location for each transition.

    Output is Sigmoid() × window_samples, constraining predictions to the
    valid range [0, window_samples].

    Output : (B, num_transitions)   ← absolute sample indices within window
    """

    def __init__(self, in_features: int, num_transitions: int,
                 window_samples: int, dropout: float):
        super().__init__()
        self.window_samples = float(window_samples)
        self.net = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.BatchNorm1d(in_features // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_features // 2, num_transitions),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * self.window_samples


# ─────────────────────────────────────────────────────────────────────────────
# FULL MODEL
# ─────────────────────────────────────────────────────────────────────────────

class RespiratoryDetector(nn.Module):
    """
    Dual Classifier-Regressor for multimodal respiratory phase detection.

    Architecture overview
    ---------------------
    Three independent encoders:
        primary_encoder   ← filtered PPG
        support_encoder   ← filtered ECG
        feature_encoder   ← Gaussian R-peak indicator

    Two parallel processing pathways:

        [Fusion pathway]
            Concatenate all three encoder outputs (dim=1)
            → fusion_core → ClassificationHead  (logits → prob_fusion)
                          → RegressionHead      (location)

        [Primary-only pathway]
            primary encoder output only
            → primary_core → ClassificationHead (logits → prob_primary)
            → RegressionHead (training only, improves encoder representations)

    Final probability = sigmoid( (logits_fusion + logits_primary) / 2 )
    Final location    = fusion regression head output

    The primary-only pathway forces the model to learn from mechanical
    (optical) information alone, making the system robust when ECG is
    corrupted by motion artefacts (which is common in WESAD wrist data).

    Forward inputs
    --------------
    primary    : (B, 1, W)   — normalised PPG window
    auxiliary  : (B, 1, W)   — normalised ECG window
    aux_marker1: (B, 1, W)   — Gaussian R-peak indicator

    Forward outputs (dict)
    ----------------------
    logits_fusion   : (B, 2)  — raw logits from fusion path
    logits_primary  : (B, 2)  — raw logits from primary-only path
    prob            : (B, 2)  — averaged, sigmoid-activated probabilities
    loc             : (B, 2)  — predicted sample indices (fusion path)
    loc_primary     : (B, 2) | None  — training-only regression (primary path)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        enc_ch = cfg.ENCODER_CH

        # ── Three independent encoders ───────────────────────────────────────
        self.primary_encoder  = Encoder(enc_ch)
        self.support_encoder  = Encoder(enc_ch)
        self.feature_encoder  = Encoder(enc_ch)

        enc_out = self.primary_encoder.out_channels  # same for all three
        fusion_in_ch = enc_out * 3                   # concat of 3 encoders

        # ── Dual core networks ───────────────────────────────────────────────
        self.fusion_core  = CoreNetwork(fusion_in_ch, cfg.CORE_CH,
                                        cfg.DENSE_UNITS, cfg.DROPOUT)
        self.primary_core = CoreNetwork(enc_out, cfg.CORE_CH,
                                        cfg.DENSE_UNITS, cfg.DROPOUT)

        # ── Output heads ─────────────────────────────────────────────────────
        # Fusion path
        self.cls_fusion = ClassificationHead(
            cfg.DENSE_UNITS, cfg.NUM_TRANSITIONS, cfg.DROPOUT)
        self.reg_fusion = RegressionHead(
            cfg.DENSE_UNITS, cfg.NUM_TRANSITIONS,
            cfg.WINDOW_SAMPLES, cfg.DROPOUT)

        # Primary-only path
        self.cls_primary = ClassificationHead(
            cfg.DENSE_UNITS, cfg.NUM_TRANSITIONS, cfg.DROPOUT)
        # Training-only regression on primary path
        self.reg_primary = RegressionHead(
            cfg.DENSE_UNITS, cfg.NUM_TRANSITIONS,
            cfg.WINDOW_SAMPLES, cfg.DROPOUT)

        # Weight initialisation
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                    nonlinearity="relu")
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d,)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        primary:     torch.Tensor,
        auxiliary:   torch.Tensor,
        aux_marker1: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:

        # ── Encode each modality independently ───────────────────────────────
        feat_ppg   = self.primary_encoder(primary)      # (B, C, T')
        feat_ecg   = self.support_encoder(auxiliary)    # (B, C, T')
        feat_rpeak = self.feature_encoder(aux_marker1)  # (B, C, T')

        # ── Primary-only pathway ─────────────────────────────────────────────
        core_primary     = self.primary_core(feat_ppg)
        logits_primary   = self.cls_primary(core_primary)         # (B, 2)
        loc_primary      = self.reg_primary(core_primary) if self.training else None

        # ── Fusion pathway ───────────────────────────────────────────────────
        fused            = torch.cat([feat_ppg, feat_ecg, feat_rpeak], dim=1)
        core_fusion      = self.fusion_core(fused)
        logits_fusion    = self.cls_fusion(core_fusion)            # (B, 2)
        loc_fusion       = self.reg_fusion(core_fusion)            # (B, 2)

        # ── Combine classification outputs ───────────────────────────────────
        # Average logits before sigmoid → equivalent to geometric mean of probs
        prob = torch.sigmoid((logits_fusion + logits_primary) / 2.0)

        return {
            "logits_fusion":  logits_fusion,
            "logits_primary": logits_primary,
            "prob":           prob,          # used during inference
            "loc":            loc_fusion,
            "loc_primary":    loc_primary,   # None during inference
        }
