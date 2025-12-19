import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram


class FeatureMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))
        return loss * 2


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, disc_outputs):
        loss = 0
        for dg in disc_outputs:
            for l in dg:
                loss += torch.mean((l - 1) ** 2)
        return loss


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, disc_outputs_r, disc_outputs_g):
        loss = 0
        for dr, dg in zip(disc_outputs_r, disc_outputs_g):
            for r, g in zip(dr, dg):
                r_loss = torch.mean((r - 1) ** 2)
                g_loss = torch.mean(g**2)
                loss += r_loss + g_loss
        return loss


class MelSpectrogramLoss(nn.Module):
    def __init__(self, sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80):
        super().__init__()
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=True,
        )

    def forward(self, y_hat, y):
        mel_hat = self.mel_spectrogram(y_hat.squeeze(1))
        mel = self.mel_spectrogram(y.squeeze(1))
        loss = F.l1_loss(mel_hat, mel)
        return loss


class HiFiGANLoss(nn.Module):
    def __init__(self, mel_lambda=45, fm_lambda=2):
        super().__init__()
        self.mel_lambda = mel_lambda
        self.fm_lambda = fm_lambda
        self.g_loss = GeneratorLoss()
        self.d_loss = DiscriminatorLoss()
        self.fm_loss = FeatureMatchingLoss()
        self.mel_loss = MelSpectrogramLoss()

    def forward(
        self,
        y,
        y_hat,
        mpd_out_real,
        mpd_out_fake,
        mpd_feats_real,
        mpd_feats_fake,
        msd_out_real,
        msd_out_fake,
        msd_feats_real,
        msd_feats_fake,
        **batch
    ):
        # Generator Loss
        loss_mel = self.mel_loss(y_hat, y)
        loss_fm = self.fm_loss(mpd_feats_real, mpd_feats_fake) + self.fm_loss(
            msd_feats_real, msd_feats_fake
        )
        loss_g_mpd = self.g_loss(mpd_out_fake)
        loss_g_msd = self.g_loss(msd_out_fake)
        loss_g = loss_g_mpd + loss_g_msd + self.fm_lambda * loss_fm + self.mel_lambda * loss_mel

        # Discriminator Loss
        loss_d_mpd = self.d_loss(mpd_out_real, mpd_out_fake)
        loss_d_msd = self.d_loss(msd_out_real, msd_out_fake)
        loss_d = loss_d_mpd + loss_d_msd

        return {
            "loss_g": loss_g,
            "loss_d": loss_d,
            "loss_mel": loss_mel,
            "loss_fm": loss_fm,
            "loss_g_mpd": loss_g_mpd,
            "loss_g_msd": loss_g_msd,
            "loss_d_mpd": loss_d_mpd,
            "loss_d_msd": loss_d_msd,
        }