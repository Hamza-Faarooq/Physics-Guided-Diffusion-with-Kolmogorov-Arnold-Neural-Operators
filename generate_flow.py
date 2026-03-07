import torch

from models.diffusion_unet import UNet
from models.pikan_fno_model import PIKANFNO

from diffusion.schedule import linear_beta_schedule, get_alphas
from diffusion.sample import p_sample

from physics.correction import physics_correct

from utils.grid import build_grid


def generate_flow():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    diffusion_model = UNet().to(device)

    physics_model = PIKANFNO().to(device)

    diffusion_model.eval()

    physics_model.eval()

    T = 1000

    betas = linear_beta_schedule(T).to(device)

    alphas,alphas_cumprod = get_alphas(betas)

    x = torch.randn(1,3,128,128).to(device)

    for t in reversed(range(T)):

        x = p_sample(
            diffusion_model,
            x,
            t,
            betas,
            alphas,
            alphas_cumprod
        )

    coords = build_grid(128,device)

    corrected = physics_correct(
        physics_model,
        x,
        coords
    )

    return corrected


if __name__ == "__main__":

    flow = generate_flow()

    print("Generated flow field shape:",flow.shape)
