import torch


def linear_beta_schedule(T):

    beta_start = 1e-4
    beta_end = 0.02

    return torch.linspace(beta_start, beta_end, T)


def get_alphas(betas):

    alphas = 1.0 - betas

    alphas_cumprod = torch.cumprod(alphas, dim=0)

    return alphas, alphas_cumprod
