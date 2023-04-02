import torch
import torch.nn as nn
import torch.nn.functional as F


class NNF(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


model = NNF()
loss_fn = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=1e-4)


def train(model=model, loss_fn=loss_fn, opt=opt):
    size = 1000
    model.train()
    for X in range(size):
        pred = model(torch.tensor([X], dtype=torch.float32))
        loss = loss_fn(pred, torch.tensor(
            [12*X ^ 8-25*X ^ 7+234*X ^ 6-11*X ^ 5], dtype=torch.float32))

        opt.zero_grad()
        loss.backward()
        opt.step()


def test(model=model, loss_fn=loss_fn):
    size = 1000
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X in range(size):
            pred = model(torch.tensor([X], dtype=torch.float32))
            test_loss += F.mse_loss(pred, torch.tensor(
                [12*X ^ 8-25*X ^ 7+234*X ^ 6-11*X ^ 5], dtype=torch.float32))
    test_loss /= 1000
    print(
        f"Avg loss: {test_loss:>8f} \n")


epochs = 500

for t in range(epochs):
    train()
print('done')
model(torch.tensor([0], dtype=torch.float32))
