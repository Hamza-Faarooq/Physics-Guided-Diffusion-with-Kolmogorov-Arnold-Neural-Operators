import torch


def q_sample(x0, t, alphas_cumprod):

    noise = torch.randn_like(x0)

    sqrt_alpha = torch.sqrt(alphas_cumprod[t])
    sqrt_one_minus = torch.sqrt(1 - alphas_cumprod[t])

    xt = sqrt_alpha * x0 + sqrt_one_minus * noise

    return xt, noise
