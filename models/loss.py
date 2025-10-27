"""
Conditional Loss for outcome generation.

This module provides loss computation for conditional generation tasks where:
- Only continuous outcomes need to be modeled (no categorical outcomes)
- Loss is computed only for masked (to-be-predicted) outcomes
- Uses diffusion-based loss for continuous values
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


# Diffusion parameters (same as original DIME)
randn_like = torch.randn_like
SIGMA_MIN = 0.002
SIGMA_MAX = 80
rho = 7
S_churn = 1
S_min = 0
S_max = float('inf')
S_noise = 1

class OutcomeLoss(nn.Module):
    """
    Loss module for conditional outcome generation.

    Key features:
    - Only computes diffusion loss for continuous outcomes
    - Supports masking: only compute loss for masked (to-be-predicted) outcomes
    - Each outcome dimension has its own predictor network
    """
    def __init__(self, n_outcome, hid_dim, dim_t=1024, dropout_rate=0.3):
        super().__init__()
        self.n_outcome = n_outcome
        self.hid_dim = hid_dim

        # Predictor networks: transform hidden states to outcome space
        self.outcome_predictors = nn.ModuleList()
        for _ in range(n_outcome):
            self.outcome_predictors.append(nn.Sequential(
                nn.Linear(hid_dim, hid_dim * 16),
                nn.ReLU(),
                nn.Linear(hid_dim * 16, 1)  # Each outcome is 1-dimensional (scalar)
            ))

        # Diffusion loss modules: one per outcome
        self.outcome_losses = nn.ModuleList()
        for _ in range(n_outcome):
            self.outcome_losses.append(
                DiffLoss(d_in=1, dim_t=dim_t, dropout_rate=dropout_rate)
            )

    def forward(self, z_outcome, gt_outcome, mask):
        B = z_outcome.shape[0]
        losses = []

        for i in range(self.n_outcome):
            z_i = z_outcome[:, i, :]  # [B, hid_dim]
            gt_i = gt_outcome[:, i:i+1]  # [B, 1]
            mask_i = mask[:, i]  # [B]

            # Predict outcome value (for conditioning the diffusion model)
            pred_i = self.outcome_predictors[i](z_i)  # [B, 1]

            # Compute diffusion loss
            loss_i = self.outcome_losses[i](gt_i, pred_i)  # [B, 1]
            loss_i = loss_i.squeeze(1)  # [B]

            # Only count loss for masked positions
            if mask_i.sum() > 0:
                loss_i_masked = (loss_i * mask_i).sum() / mask_i.sum()
            else:
                loss_i_masked = torch.tensor(0.0, device=z_outcome.device)

            losses.append(loss_i_masked)

        # Average loss across all outcomes
        total_loss = sum(losses) / len(losses)
        loss_per_outcome = torch.stack(losses)

        return total_loss, loss_per_outcome

    def sample(self, z_outcome, num_steps=50, device='cuda'):
        B = z_outcome.shape[0]
        sampled_outcomes = []

        with torch.no_grad():
            for i in range(self.n_outcome):
                z_i = z_outcome[:, i, :]  # [B, hid_dim]

                # Predict outcome (conditioning for diffusion)
                pred_i = self.outcome_predictors[i](z_i)  # [B, 1]

                # Sample using diffusion model
                sampled_i = self.outcome_losses[i].sample(
                    B, 1, pred_i, num_steps, device
                )  # [B, 1]

                sampled_outcomes.append(sampled_i)

        sampled_outcomes = torch.cat(sampled_outcomes, dim=1)  # [B, n_outcome]
        return sampled_outcomes

    def sample_single_outcome(self, z_outcome_i, outcome_idx, num_steps=50, device='cuda'):
        B = z_outcome_i.shape[0]

        with torch.no_grad():
            # Predict outcome
            pred_i = self.outcome_predictors[outcome_idx](z_outcome_i)  # [B, 1]

            # Sample using diffusion model
            sampled_i = self.outcome_losses[outcome_idx].sample(
                B, 1, pred_i, num_steps, device
            )  # [B, 1]

        return sampled_i



class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class MLPDiffusion(nn.Module):
    """MLP-based denoising network for diffusion model."""
    def __init__(self, d_in, dim_t=512):
        super().__init__()
        self.dim_t = dim_t

        self.proj = nn.Linear(d_in, dim_t)

        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )

        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

        self.z_embed = nn.Sequential(
            nn.Linear(d_in, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

    def forward(self, x, noise_labels, z):
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
        emb = self.time_embed(emb)

        emb += self.z_embed(z)

        x = self.proj(x) + emb
        return self.mlp(x)


class DiffLoss(nn.Module):
    """
    Conditional Diffusion Loss for continuous outcomes.

    Same as the original DiffLoss but designed for conditional generation.
    """
    def __init__(
        self,
        d_in,
        dim_t,
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
        gamma=5,
        sigma_min=0,
        sigma_max=float('inf'),
        dropout_rate=0.3
    ):
        super().__init__()

        self.denoise_fn = MLPDiffusion(d_in, dim_t)
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.gamma = gamma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def precond(self, denoise_fn, x, z, sigma):
        """Preconditioning for the denoising network."""
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1)
        dtype = torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        x_in = c_in * x
        F_x = denoise_fn(x_in, c_noise.flatten(), z)

        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def forward(self, data, z):
        """
        Compute diffusion loss.

        Args:
            data: [B, d_in] ground truth data
            z: [B, d_in] conditioning information (from transformer)

        Returns:
            loss: [B, d_in] per-sample, per-dimension loss (NOT reduced)
        """
        rnd_normal = torch.randn(data.shape[0], device=data.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        y = data
        n = torch.randn_like(y) * sigma.unsqueeze(1)
        D_yn = self.precond(self.denoise_fn, y + n, z, sigma)

        target = y
        loss = weight.unsqueeze(1) * ((D_yn - target) ** 2)

        return loss

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def sample_step(self, z, num_steps, i, t_cur, t_next, x_next):
        """Single sampling step using Heun's method."""
        x_cur = x_next
        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = self.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = self.precond(self.denoise_fn, x_hat, z, t_hat.expand(x_next.shape[0], 1)).to(torch.float32)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = self.precond(self.denoise_fn, x_next, z, t_next.expand(x_next.shape[0], 1)).to(torch.float32)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    def sample(self, B, embed_dim, z, num_steps=50, device='cuda'):
        """
        Unconditional sampling from the diffusion model.

        Args:
            B: batch size
            embed_dim: dimension of the outcome
            z: [B, embed_dim] conditioning information
            num_steps: number of diffusion steps
            device: torch device

        Returns:
            x_next: [B, embed_dim] sampled outcomes
        """
        latents = torch.randn([B, embed_dim], device=device)
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)

        sigma_min = max(SIGMA_MIN, self.sigma_min)
        sigma_max = min(SIGMA_MAX, self.sigma_max)

        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                    sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        x_next = latents.to(torch.float32) * t_steps[0]

        with torch.no_grad():
            for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
                x_next = self.sample_step(z, num_steps, i, t_cur, t_next, x_next)

        return x_next
