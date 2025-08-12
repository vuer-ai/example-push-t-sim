from time import sleep

import torch
from scipy.stats.tests.test_continuous_fit_censored import optimizer

from lucidxr.learning.models.detr_vae import SimpleMLP

example_obs = torch.rand((1, 22))
example_action = torch.rand((1, 1, 3))
model = SimpleMLP(22, 3)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
loss_fn = torch.nn.L1Loss()

def train_loop(num_epochs):
    for t in range(num_epochs):
        # Compute prediction and loss
        pred, _= model(example_obs, example_action)
        loss = loss_fn(pred, example_action)
        l1_max = torch.abs(pred - example_action).max()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss.item()
        print(f"epoch {t}: loss: {loss}, l1_max: {l1_max}")

        sleep(0.01)

train_loop(10000)

