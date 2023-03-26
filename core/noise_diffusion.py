import torch as pt
import math


class NoisifyImage:
    def __init__(self, t_max=1000, device="cuda", noise_schedule="cosine", **kwargs):
        if noise_schedule == "cosine":
            alph_bars, betas = self._get_cosine_noise_schedule(
                t_max=t_max, s=kwargs["s"], beta_max=kwargs["beta_max"]
            )
            self.alph_bars = alph_bars.to(device)
            self.betas = betas.to(device)
        elif noise_schedule == "linear":
            alph_bars, betas = self._get_linear_noise_schedule(
                t_max=t_max, beta_min=kwargs["beta_min"], beta_max=kwargs["beta_max"]
            )
            self.alph_bars = alph_bars.to(device)
            self.betas = betas.to(device)
        else:
            raise ValueError

    @staticmethod
    def _get_linear_noise_schedule(t_max=1000, beta_min=1e-4, beta_max=0.02):
        betas = pt.linspace(beta_min, beta_max, t_max)
        alpha_bars = pt.cumprod(1 - betas, dim=0)
        return alpha_bars, betas

    @staticmethod
    def _get_cosine_noise_schedule(t_max=1000, s=0.008, beta_max=0.999):
        t_frac = pt.linspace(0, 1, t_max + 1)
        f = pt.cos((t_frac + s) * math.pi / 2 / (1 + s)) ** 2
        alph_bar = f / f[0]
        beta = pt.minimum(1 - alph_bar[1:] / alph_bar[:-1], pt.tensor([beta_max]))
        return alph_bar[1:], beta

    def noisify_to_t(self, imgs, t):
        alph_bar_t = self.alph_bars[t][:, None, None, None]
        noise = pt.randn_like(imgs)
        noisy_imgs = pt.sqrt(alph_bar_t) * imgs + pt.sqrt(1 - alph_bar_t) * noise
        return noisy_imgs, noise

