import torch
import torch.optim as optim

from models.diffusion_unet import UNet
from diffusion.schedule import linear_beta_schedule, get_alphas
from diffusion.forward import q_sample


def train_diffusion(epochs=200):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    T = 1000

    betas = linear_beta_schedule(T).to(device)

    alphas, alphas_cumprod = get_alphas(betas)

    alphas = alphas.to(device)

    alphas_cumprod = alphas_cumprod.to(device)

    for epoch in range(epochs):

        flow = torch.randn(1, 3, 128, 128).to(device)

        t = torch.randint(0, T, (1,), device=device)

        xt, noise = q_sample(flow, t, alphas_cumprod)

        pred_noise = model(xt)

        loss = torch.mean((noise - pred_noise) ** 2)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if epoch % 10 == 0:

            print(f"Epoch {epoch} | Diffusion Loss {loss.item():.6f}")

    torch.save(model.state_dict(), "diffusion_model.pt")

    return model


if __name__ == "__main__":

    train_diffusion()
