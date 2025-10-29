"""Lightweight models for the AstroCLIP teaching pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


def _weight_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ImageAutoencoder(L.LightningModule):
    """Simple convolutional autoencoder for 144x144 RGB images."""

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_channels: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(inplace=True),
        )

        self.encoder_fc = nn.Sequential(
            nn.Linear(hidden_channels * 4 * 18 * 18, hidden_channels * 8),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels * 8, embed_dim),
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_channels * 8),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels * 8, hidden_channels * 4 * 18 * 18),
            nn.ReLU(inplace=True),
        )

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        self.apply(_weight_init)

    @property
    def embed_dim(self) -> int:
        return self.hparams.embed_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_cnn(x)
        h = h.view(h.size(0), -1)
        return self.encoder_fc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        h = h.view(h.size(0), -1, 18, 18)
        return self.decoder_cnn(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        images = batch
        recon = self(images)
        loss = F.mse_loss(recon, images)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        images = batch
        recon = self(images)
        loss = F.mse_loss(recon, images)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class SpectrumAutoencoder(L.LightningModule):
    """Compact fully-connected autoencoder for 1D spectra."""

    def __init__(
        self,
        input_dim: int = 7781,
        embed_dim: int = 256,
        hidden_dim: int = 1024,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, embed_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )

        self.apply(_weight_init)

    @property
    def embed_dim(self) -> int:
        return self.hparams.embed_dim

    def encode(self, spectrum: torch.Tensor) -> torch.Tensor:
        return self.encoder(spectrum)

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.decoder(embedding)

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(spectrum))

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        spectra = batch
        recon = self(spectra)
        loss = F.mse_loss(recon, spectra)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        spectra = batch
        recon = self(spectra)
        loss = F.mse_loss(recon, spectra)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class SpectrumTransformer(L.LightningModule):
    """Lightweight transformer encoder that embeds spectra into a fixed representation."""

    def __init__(
        self,
        input_dim: int = 7781,
        patch_size: int = 16,
        embed_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if input_dim % patch_size != 0:
            raise ValueError("input_dim must be divisible by patch_size")

        num_patches = input_dim // patch_size
        self.patch_embed = nn.Linear(patch_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=embed_dim * 2,
            activation="gelu",
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.reconstruction = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, input_dim),
        )

        self.apply(_weight_init)

    @property
    def embed_dim(self) -> int:
        return self.hparams.embed_dim

    def _patchify(self, spectrum: torch.Tensor) -> torch.Tensor:
        b, seq_len = spectrum.shape
        ps = self.hparams.patch_size
        patches = spectrum.view(b, seq_len // ps, ps)
        return patches

    def encode(self, spectrum: torch.Tensor) -> torch.Tensor:
        patches = self._patchify(spectrum)
        tokens = self.patch_embed(patches) + self.positional_encoding
        encoded = self.transformer(tokens)
        pooled = encoded.mean(dim=1)
        return self.norm(pooled)

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        return self.encode(spectrum)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        spectra = batch
        emb = self.encode(spectra)
        recon = self.reconstruction(emb)
        loss = F.mse_loss(recon, spectra)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        spectra = batch
        emb = self.encode(spectra)
        recon = self.reconstruction(emb)
        loss = F.mse_loss(recon, spectra)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class SmallCLIPModel(L.LightningModule):
    """CLIP-style alignment using pretrained image and spectrum encoders."""

    def __init__(
        self,
        image_encoder: ImageAutoencoder,
        spectrum_encoder: SpectrumAutoencoder,
        projection_dim: Optional[int] = None,
        lr: float = 5e-4,
        weight_decay: float = 1e-5,
        temperature: float = 0.07,
        finetune_encoders: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["image_encoder", "spectrum_encoder"])

        self.image_encoder = image_encoder
        self.spectrum_encoder = spectrum_encoder

        img_dim = image_encoder.embed_dim
        sp_dim = spectrum_encoder.embed_dim

        projection_dim = projection_dim or max(img_dim, sp_dim)
        self.image_projection = nn.Linear(img_dim, projection_dim)
        self.spectrum_projection = nn.Linear(sp_dim, projection_dim)

        self.logit_scale = nn.Parameter(torch.tensor(1 / temperature).log())

        if not finetune_encoders:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            for param in self.spectrum_encoder.parameters():
                param.requires_grad = False

        self.apply(_weight_init)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.hparams.finetune_encoders):
            z = self.image_encoder.encode(images)
        return self.image_projection(z)

    def encode_spectrum(self, spectra: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.hparams.finetune_encoders):
            z = self.spectrum_encoder.encode(spectra)
        return self.spectrum_projection(z)

    def forward(
        self, images: torch.Tensor, spectra: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_features = F.normalize(self.encode_image(images), dim=-1)
        spec_features = F.normalize(self.encode_spectrum(spectra), dim=-1)
        return img_features, spec_features

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        spectra = batch["spectrum"]
        img_features, spec_features = self(images, spectra)

        logits = img_features @ spec_features.t()
        logits = logits * self.logit_scale.exp()

        targets = torch.arange(logits.size(0), device=self.device)
        loss_i = F.cross_entropy(logits, targets)
        loss_t = F.cross_entropy(logits.t(), targets)
        loss = (loss_i + loss_t) / 2

        self.log_dict(
            {"train_loss": loss, "logit_scale": self.logit_scale.exp()},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        images = batch["image"]
        spectra = batch["spectrum"]
        img_features, spec_features = self(images, spectra)

        logits = img_features @ spec_features.t()
        logits = logits * self.logit_scale.exp()

        targets = torch.arange(logits.size(0), device=self.device)
        loss_i = F.cross_entropy(logits, targets)
        loss_t = F.cross_entropy(logits.t(), targets)
        loss = (loss_i + loss_t) / 2

        self.log_dict(
            {"val_loss": loss, "logit_scale": self.logit_scale.exp()},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self):
        parameters = [
            {"params": self.image_projection.parameters()},
            {"params": self.spectrum_projection.parameters()},
            {"params": [self.logit_scale]},
        ]
        if self.hparams.finetune_encoders:
            parameters.append({"params": self.image_encoder.parameters()})
            parameters.append({"params": self.spectrum_encoder.parameters()})

        optimizer = torch.optim.AdamW(
            parameters, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
