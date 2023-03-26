import torch as pt

from core.noise_diffusion import NoisifyImage


def test_calculated_linear_noise_params_have_correct_shape():
    length = 1000
    noisify = NoisifyImage(
        device="cpu", noise_schedule="linear", beta_min=1e-4, beta_max=0.02,
        t_max=length
    )
    alph_bars = noisify.alph_bars
    betas = noisify.betas

    assert alph_bars.shape == (length, )
    assert betas.shape == (length, )


def test_calculated_linear_noise_params_are_correct():
    beta_length = 1000
    beta_min = 1e-4
    beta_max = 0.02

    noisify = NoisifyImage(
        device="cpu", noise_schedule="linear", beta_min=beta_min, beta_max=beta_max,
        t_max=beta_length
    )
    alph_bars = noisify.alph_bars
    betas = noisify.betas

    expected_betas = pt.linspace(beta_min, beta_max, beta_length)
    expected_alph_bars = pt.cumprod(1-expected_betas, dim=0)

    pt.testing.assert_close(alph_bars, expected_alph_bars)
    pt.testing.assert_close(betas, expected_betas)


def test_linear_noisify_is_correct():
    imgs = pt.tensor([
        [[
            [-1., 0.5],
            [-0.5, 1.]
        ]] * 3,
        [[
            [1., -0.5],
            [0.5, -1.]
        ]] * 3,
    ])
    ts = pt.tensor([20, 102])

    noisify = NoisifyImage(
        device="cpu", noise_schedule="linear", beta_min=1e-4, beta_max=0.02,
        t_max=1000
    )
    noisy_imgs, noise = noisify.noisify_to_t(imgs, ts)

    ## Expected img_0
    alph_bar_0 = noisify.alph_bars[ts[0]]
    expected_img_0 = pt.sqrt(alph_bar_0) * imgs[0] + pt.sqrt(1 - alph_bar_0) * noise[0]

    pt.testing.assert_close(noisy_imgs[0], expected_img_0)

    ## Expected img_1
    alph_bar_1 = noisify.alph_bars[ts[1]]
    expected_img_1 = pt.sqrt(alph_bar_1) * imgs[1] + pt.sqrt(1 - alph_bar_1) * noise[1]

    pt.testing.assert_close(noisy_imgs[1], expected_img_1)
