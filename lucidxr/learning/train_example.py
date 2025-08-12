import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from lucidxr.learning.models.detr_vae import SimpleMLP

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

training_data, _ = torch.utils.data.random_split(training_data, [1000, 59000])
test_data, _ = torch.utils.data.random_split(test_data, [1000, 9000])

learning_rate = 1e-3
batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)
model = SimpleMLP(28*28, 10)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    test_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.flatten(start_dim=2).squeeze()
        one_hot_labels = F.one_hot(y, num_classes=10)
        one_hot_labels = one_hot_labels.unsqueeze(1).float()
        pred= model(X, one_hot_labels).squeeze()
        loss = loss_fn(pred, one_hot_labels.squeeze())
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X = X.flatten(start_dim=2).squeeze()
            one_hot_labels = F.one_hot(y, num_classes=10)
            one_hot_labels = one_hot_labels.unsqueeze(1).float()
            pred = model(X, one_hot_labels).squeeze()
            test_loss += loss_fn(pred, one_hot_labels.squeeze()).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss()

n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("number of parameters: %.2fM" % (n_parameters / 1e6,))

no_grad_params = [p for p in model.parameters() if not p.requires_grad]
grad_params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.AdamW(
    [
        {"params": no_grad_params},
        {"params": grad_params, "lr": learning_rate},
    ],
    lr=learning_rate
)

epochs = 1000
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(train_dataloader, model, loss_fn)
print("Done!")