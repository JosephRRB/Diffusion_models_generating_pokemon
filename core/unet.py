import torch as pt
import torch.nn as nn
import math

class BottleneckConvBlock(nn.Module):
    def __init__(self, in_c, out_c, t_emb_dim):
        super().__init__()
        middle_c = max(int(in_c // 2), 16)
        self.t_emb_nn_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, middle_c)
        )
        self.in_layers = nn.Sequential(
            nn.Conv2d(
                in_c, middle_c, kernel_size=1, padding=0, bias=False
            ),
            nn.GroupNorm(1, middle_c),
            nn.GELU(),
            nn.Conv2d(
                middle_c, middle_c, kernel_size=3, padding=1, bias=False
            )
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(1, middle_c),
            nn.GELU(),
            nn.Conv2d(
                middle_c, middle_c, kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(1, middle_c),
            nn.GELU(),
            nn.Conv2d(
                middle_c, out_c, kernel_size=1, padding=0, bias=False
            ),
        )

    def forward(self, img, t_emb):
        hidden = self.in_layers(img)
        t_emb = self.t_emb_nn_layers(t_emb)[:, :, None, None]
        return self.out_layers(hidden + t_emb)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_c, out_c, t_emb_dim, mode=None):
        super().__init__()
        shortcut_blocks = [nn.Identity()]

        pre_conv_blocks = [nn.GroupNorm(1, in_c), nn.GELU()]
        if mode == "upsample":
            pre_conv_blocks += [nn.Upsample(scale_factor=2, mode="nearest")]
            shortcut_blocks += [nn.Upsample(scale_factor=2, mode="nearest")]

        self.pre_conv = nn.Sequential(*pre_conv_blocks)
        self.conv = BottleneckConvBlock(in_c, out_c, t_emb_dim)
        if in_c != out_c:
            shortcut_blocks += [
                nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, bias=False)
            ]

        post_conv_blocks = [nn.Identity()]
        if mode == "downsample":
            post_conv_blocks += [nn.AvgPool2d(2)]
            shortcut_blocks += [nn.AvgPool2d(2)]
        self.post_conv = nn.Sequential(*post_conv_blocks)

        self.shortcut = nn.Sequential(*shortcut_blocks)

    def forward(self, img, t_emb):
        hidden = self.pre_conv(img)
        hidden = self.conv(hidden, t_emb)
        return self.post_conv(hidden) + self.shortcut(img)


class SelfAttentionBlock(nn.Module):
    def __init__(self, img_c, img_size):
        super().__init__()
        self.img_c = img_c
        self.img_size = img_size
        self.self_attention = nn.TransformerEncoderLayer(
            self.img_c, nhead=4, batch_first=True, activation="gelu",
            dim_feedforward=self.img_c, dropout=0.0, norm_first=True
        )

    def forward(self, x):
        x = x.view(-1, self.img_c, self.img_size * self.img_size).swapaxes(1, 2)
        x = self.self_attention(x)
        x = x.swapaxes(2, 1).view(-1, self.img_c, self.img_size, self.img_size)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, in_c, out_c, t_emb_dim, in_img_size):
        super().__init__()
        self.conv = ResidualConvBlock(in_c, out_c, t_emb_dim)
        self.attention = SelfAttentionBlock(out_c, in_img_size)
        self.downsample = ResidualConvBlock(
            out_c, out_c, t_emb_dim, mode="downsample"
        )

    def forward(self, img, t_emb):
        hidden = self.conv(img, t_emb)
        hidden = self.attention(hidden)
        return self.downsample(hidden, t_emb)


class MiddleLayer(nn.Module):
    def __init__(self, img_c, t_emb_dim, img_size):
        super().__init__()
        self.conv1 = ResidualConvBlock(img_c, img_c, t_emb_dim)
        self.attention = SelfAttentionBlock(img_c, img_size)
        self.conv2 = ResidualConvBlock(img_c, img_c, t_emb_dim)

    def forward(self, img, t_emb):
        hidden = self.conv1(img, t_emb)
        hidden = self.attention(hidden)
        return self.conv2(hidden, t_emb)


class DecoderLayer(nn.Module):
    def __init__(self, in_c, out_c, t_emb_dim, in_img_size):
        super().__init__()
        concat_c = 2 * in_c
        upsampled_size = 2 * in_img_size
        self.upsample = ResidualConvBlock(
            concat_c, in_c, t_emb_dim, mode="upsample"
        )
        self.attention = SelfAttentionBlock(in_c, upsampled_size)
        self.conv = ResidualConvBlock(in_c, out_c, t_emb_dim)

    def forward(self, img, skip_img, t_emb):
        assert img.shape == skip_img.shape
        imgs = pt.cat([img, skip_img], dim=1)
        hidden = self.upsample(imgs, t_emb)
        hidden = self.attention(hidden)
        return self.conv(hidden, t_emb)


class UNet3x64x64(nn.Module):
    def __init__(self, t_emb_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.t_emb_dim = t_emb_dim
        img_sizes = [64, 32, 16, 8, 4]
        channels = [16, 32, 64, 128, 256]
        enc_in_cs = channels[:-1]
        enc_out_cs = channels[1:]

        self.in_conv = nn.Conv2d(
            3, enc_in_cs[0], kernel_size=3, padding=1, bias=False
        ).to(self.device)

        self.encoder_layers = [
            EncoderLayer(in_c, out_c, self.t_emb_dim, in_img_size).to(self.device)
            for in_c, out_c, in_img_size in zip(
                enc_in_cs, enc_out_cs, img_sizes[:-1]
            )
        ]

        self.mid_layer = MiddleLayer(
            img_c=enc_out_cs[-1], t_emb_dim=self.t_emb_dim, img_size=img_sizes[-1]
        ).to(self.device)

        self.decoder_layers = [
            DecoderLayer(in_c, out_c, self.t_emb_dim, in_img_size).to(self.device)
            for in_c, out_c, in_img_size in zip(
                enc_out_cs[::-1], enc_in_cs[::-1], img_sizes[::-1][:-1]
            )
        ]

        self.out_conv = nn.Sequential(
            nn.GroupNorm(1, 2 * enc_in_cs[0]),
            nn.GELU(),
            nn.Conv2d(2 * enc_in_cs[0], 3, kernel_size=3, padding=1, bias=False)
        ).to(self.device)

    def forward(self, imgs, ts):
        t_embs = self._embed_time_step(ts, embedding_dim=self.t_emb_dim)

        hs = [self.in_conv(imgs)]
        for encoder in self.encoder_layers:
            hs.append(
                encoder(hs[-1], t_embs)
            )

        y = self.mid_layer(hs[-1], t_embs)

        for decoder in self.decoder_layers:
            s = hs.pop()
            y = decoder(y, s, t_embs)

        return self.out_conv(pt.cat([y, hs[-1]], dim=1))

    def _embed_time_step(self, ts, embedding_dim=256, n=10000):
        half_dim = embedding_dim // 2
        log_n = math.log(n)

        sin_emb_inds = pt.arange(0, half_dim, dtype=pt.float)
        cos_emb_inds = pt.arange(0, half_dim + embedding_dim % 2, dtype=pt.float)

        sin_freqs = pt.exp(-log_n * sin_emb_inds / half_dim).to(self.device)
        cos_freqs = pt.exp(-log_n * cos_emb_inds / half_dim).to(self.device)

        sin_embs = pt.sin(sin_freqs * ts)
        cos_embs = pt.cos(cos_freqs * ts)

        t_embs = pt.cat([sin_embs, cos_embs], dim=-1)
        return t_embs

