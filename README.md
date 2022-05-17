# Classifier Divergence
This is a package for computing the divergence between a source sample S 
and target sample T. Specifically, this package computes
a model-dependent divergence, sometimes referred to as the h-discrepancy or h\Delta\H-divergence
computed for a model h by finding g which maximizes 

D(g) = max(Pr(h(x) != g(x) | x ~ S) - Pr(h(x) != g(x) | x ~ T),
    Pr(h(x) != g(x) | x ~ S) - Pr(h(x) != g(x) | x ~ T)).

## Background
The properties of this statistic are well studied in the papers:
[The Change that Matters in Discourse Parsing: Estimating the Impact of Domain Shift on Parser Error](https://arxiv.org/abs/2203.11317) to appear 
in Findings of [ACL 2022](https://www.2022.aclweb.org) and “PAC-Bayesian Domain Adaptation Bounds for Multiclass Learners” to appear in 
[UAI 2022](https://www.auai.org/uai2022/). The former studies the bias of this statistic as an estimator for parser error, while the latter studies 
sample-complexity in domain adaptation, proposing a number of error bounds which contain this statistic as a key term.

Code for this package was derived from code used in the aforementioned papers. Please, consider citing these papers if you use this package.

## Relevant Links
arXiv (ACL 2022): https://arxiv.org/abs/2203.11317

arXiv (UAI 2022): Forthcoming

shared code: https://github.com/anthonysicilia/multiclass-domain-divergence

UAI code: https://github.com/anthonysicilia/pacbayes-adaptation-UAI2022

ACL code: https://github.com/anthonysicilia/change-that-matters-ACL2022

# Installation
To install this package, run the following command: <br>
``` pip install git+https://github.com/anthonysicilia/classifier-divergence/edit/main/README.md ```

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
