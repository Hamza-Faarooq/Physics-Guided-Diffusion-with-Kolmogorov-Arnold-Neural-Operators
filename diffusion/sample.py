import torch


def p_sample(model, x, t, betas, alphas, alphas_cumprod):

    noise_pred = model(x)

    beta = betas[t]

    sqrt_one_minus = torch.sqrt(1 - alphas_cumprod[t])
    sqrt_recip_alpha = torch.sqrt(1.0 / alphas[t])

    mean = sqrt_recip_alpha * (
        x - beta / sqrt_one_minus * noise_pred
    )

    if t > 0:

        noise = torch.randn_like(x)

    else:

        noise = torch.zeros_like(x)

    return mean + torch.sqrt(beta) * noise


def sample_flow(model, shape, T, betas, alphas, alphas_cumprod, device):

    x = torch.randn(shape).to(device)

    for t in reversed(range(T)):

        x = p_sample(model, x, t, betas, alphas, alphas_cumprod)

    return x
