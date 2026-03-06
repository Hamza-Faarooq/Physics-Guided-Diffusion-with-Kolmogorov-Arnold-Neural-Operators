import torch
import torch.optim as optim

from models.pikan_fno_model import PIKANFNO
from physics.loss import physics_loss
from training.sampler import sample_points


def train_solver(epochs=5000, batch_size=2048):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PIKANFNO().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):

        coords = sample_points(batch_size, device)

        optimizer.zero_grad()

        loss = physics_loss(model, coords)

        loss.backward()

        optimizer.step()

        if epoch % 100 == 0:

            print(
                f"Epoch {epoch} | Physics Loss {loss.item():.6f}"
            )

    torch.save(model.state_dict(), "pikan_solver.pt")

    return model


if __name__ == "__main__":

    train_solver()
