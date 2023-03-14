from tqdm import tqdm

import torch as pt

from core.noise_diffusion import NoisifyImage
from core.unet import UNet3x64x64
from core.utils import _create_image_loader


class Runner:
    def __init__(self, batch_size=2, image_size=64, t_max=1000, t_emb_dim=256, lr=0.0001):
        self.t_max = t_max
        self.image_size = image_size
        self.noisify = NoisifyImage(t_max=t_max)
        self.unet = UNet3x64x64(t_emb_dim=t_emb_dim)
        self.image_loader = _create_image_loader(batch_size=batch_size, image_size=image_size)
        self.mse_loss = pt.nn.MSELoss()
        self.optimizer = pt.optim.Adam(self.unet.parameters(), lr=lr)

    def _training_step(self):
        pbar = tqdm(self.image_loader)
        for img_batch in pbar:
            ts = pt.randint(self.t_max, (img_batch.shape[0],))
            noisy_img_batch, noise = self.noisify.noisify_to_t(img_batch, ts)
            pred_noise = self.unet(noisy_img_batch, ts.view(-1, 1))
            loss = self.mse_loss(noise, pred_noise)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pbar.set_postfix(MSE=loss.item())

    def train(self, n_epochs=10):
        for i in range(n_epochs):
            self._training_step()

    def sample(self, n_samples=2):
        self.unet.eval()
        with pt.no_grad():
            img = pt.randn((n_samples, 3, self.image_size, self.image_size))
            for t in tqdm(reversed(range(1, self.t_max))):
                ts = pt.ones(n_samples, dtype=pt.int64) * t
                pred_noise = self.unet(img, ts.view(-1, 1))
                alph_bar = self.noisify.alph_bars[ts][:, None, None, None]
                beta = self.noisify.betas[ts][:, None, None, None]

                noise = pt.randn_like(img)
                if t == 1:
                    noise = 0 * noise

                img = (img - beta * pred_noise / pt.sqrt(1 - alph_bar)) / pt.sqrt(1 - beta) + pt.sqrt(beta) * noise

        self.unet.train()
        return img
