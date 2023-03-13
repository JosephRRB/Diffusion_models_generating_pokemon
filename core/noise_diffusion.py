import torch as pt
import math


class NoisifyImage:
    def __init__(self, t_max=1000, s=0.008, beta_max=0.999):
        self.alph_bars, self.betas = self._get_cosine_noise_schedule(
            t_max=t_max, s=s, beta_max=beta_max
        )

    @staticmethod
    def _get_cosine_noise_schedule(t_max=1000, s=0.008, beta_max=0.999):
        t_frac = pt.linspace(0, 1, t_max + 1)
        f = pt.cos((t_frac + s) * math.pi / 2 / (1 + s)) ** 2
        alph_bar = f / f[0]
        beta = pt.minimum(1 - alph_bar[1:] / alph_bar[:-1], pt.tensor([beta_max]))
        return alph_bar[1:], beta

    def noisify_to_t(self, imgs, t):
        alph_bar_t = pt.gather(self.alph_bars, 0, t)[:, None, None, None]
        noise = pt.randn_like(imgs)
        noisy_imgs = pt.sqrt(alph_bar_t) * imgs + pt.sqrt(1 - alph_bar_t) * noise
        return noisy_imgs, noise

