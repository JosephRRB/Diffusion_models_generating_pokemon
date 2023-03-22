from tqdm import tqdm

import torch as pt
from torch.utils.tensorboard import SummaryWriter

from core.noise_diffusion import NoisifyImage
from core.unet import UNet3x64x64
from core.utils import _create_image_loader


class Runner:
    def __init__(self, batch_size=2, image_size=64, t_max=1000, t_emb_dim=256, lr=0.0001, device="cuda"):
        self.t_max = t_max
        self.image_size = image_size
        self.device=device
        self.noisify = NoisifyImage(t_max=t_max, device=self.device)
        self.unet = UNet3x64x64(t_emb_dim=t_emb_dim, device=self.device)
        self.image_loader = _create_image_loader(batch_size=batch_size, image_size=image_size)
        self.mse_loss = pt.nn.MSELoss()
        self.optimizer = pt.optim.Adam(self.unet.parameters(), lr=lr)
        self.logger = SummaryWriter()

    def _training_step(self):
        losses = []
        for img_batch in tqdm(self.image_loader):
            self.optimizer.zero_grad()

            ts = pt.randint(self.t_max, (img_batch.shape[0],)).to(self.device)
            noisy_img_batch, noise = self.noisify.noisify_to_t(
                img_batch.to(self.device), ts
            )
            pred_noise = self.unet(noisy_img_batch, ts.view(-1, 1))

            loss = self.mse_loss(noise, pred_noise)
            loss.backward()

            self.optimizer.step()

            losses.append(loss.item())

        return sum(losses) / len(losses)

    def train(self, n_epochs=10):
        for i in range(n_epochs):
            epoch_ave_loss = self._training_step()

            if i % 100 == 0:
                print(f"Iteration: {i}, Ave Loss of Epoch:, {epoch_ave_loss:.4f}")

            self.logger.add_scalar("Ave Loss", epoch_ave_loss, i)

        self.logger.flush()
        self.logger.close()

    def sample(self, n_samples=2):
        self.unet.eval()
        with pt.no_grad():
            img = pt.randn((n_samples, 3, self.image_size, self.image_size)).to(self.device)
            for t in tqdm(reversed(range(1, self.t_max))):
                ts = (pt.ones(n_samples, dtype=pt.int64) * t).to(self.device)
                pred_noise = self.unet(img, ts.view(-1, 1))
                alph_bar = self.noisify.alph_bars[ts][:, None, None, None]
                beta = self.noisify.betas[ts][:, None, None, None]

                noise = pt.randn_like(img)
                if t == 1:
                    noise = 0 * noise

                img = (img - beta * pred_noise / pt.sqrt(1 - alph_bar)) / pt.sqrt(1 - beta) + pt.sqrt(beta) * noise

        self.unet.train()
        return img
