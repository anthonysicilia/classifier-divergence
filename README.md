This is a package for computing the divergence between a source sample S 
and target sample T. Specifically, this package computes
a model-dependent divergence, sometimes referred to as the h-discrepancy,
computed for a model h by finding g which maximizes 

D(g) = max(Pr(h(x) = g(x) | x ~ S) - Pr(h(x) != g(x) | x ~ T),
    Pr(h(x) != g(x) | x ~ S) - Pr(h(x) = g(x) | x ~ T))


# Installation
To install this package, run the following command: <br>
``` pip install XXXX ```

# Dependencies
This package was built and tested on Python 3.7.6

# Examples
## Compute Classifier Divergence on MNIST
```python
import torch
import torchvision as tv

from tqdm import tqdm

from classifier_divergence.h_discrepancy import Discrepancy
from classifier_divergence.h_discrepancy import NoSchedule
from classifier_divergence.h_discrepancy import Default

IMAGE_MEAN = 0.1307
IMAGE_STD = 0.3081

def get_datasets():
    source_transform = tv.transforms.Compose([
        tv.transforms.Grayscale(num_output_channels=1), 
        tv.transforms.Resize((28, 28)), 
        tv.transforms.ToTensor(), 
        tv.transforms.Normalize((IMAGE_MEAN,), (IMAGE_STD,))])
    source = tv.datasets.MNIST('./', train=True, download=True, 
        transform=source_transform)
    target_transform = tv.transforms.Compose([
        tv.transforms.Grayscale(num_output_channels=1), 
        tv.transforms.Resize((28,28)), 
        tv.transforms.RandomRotation(degrees=360),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((IMAGE_MEAN,), (IMAGE_STD,))])
    target = tv.datasets.MNIST('./', train=True, download=True, 
        transform=target_transform)
    return source, target

def model_init(hidden_dim):
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, 3, 1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 64, 3, 1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Flatten(1),
        torch.nn.Linear(9216, hidden_dim),
        torch.nn.Linear(hidden_dim, 10))

def example():
    print('This should take about 10-15 minutes on a CPU.')
    stat = Discrepancy(model_init=model_init,
        model_init_params={'hidden_dim' : 128},
        batch_size=250, epochs=1, 
        optim_class=torch.optim.SGD,
        optim_params={'lr' : 1e-2, 'momentum' : 0.9},
        scheduler_class=NoSchedule,
        commands=Default,
        verbose=True)
    # NOTE: Please, check the prototype for the Commands class
    # to implement more complicated training procedures
    source, target = get_datasets()
    loss_fn = torch.nn.CrossEntropyLoss()
    model = model_init(128)
    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    source_iterator = torch.utils.data.DataLoader(source,
        batch_size=250, shuffle=True)
    for x, y in tqdm(source_iterator):
        optim.zero_grad()
        loss_fn(model(x), y).backward()
        optim.step()
    model.eval()
    stat = stat.compute(model, source, target)
    print(f'Discrepancy: {stat:.4f}')

if __name__ == '__main__':
    example()

 ```
