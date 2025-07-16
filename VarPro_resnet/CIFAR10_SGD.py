import sys
sys.path.append('..')

from VarPro import *
from resnet import *

import torchvision
import torchvision.transforms.v2 as transforms

import argparse
import time
import gzip
import pickle
import os

parser = argparse.ArgumentParser()

## Mandatory arguments
parser.add_argument('--epochs', '-e', type=int) ## Number of epochs
parser.add_argument('--batch_size', '-bs', type=int) ## Number of data samples

## Default arguments
parser.add_argument('--lmbda', '-l', type=float, default=1e-3)  ## Regularization parameter
parser.add_argument('--time_scale', '-ts', type=float, default=1e-3) ## Time scale of the gradient flow
parser.add_argument('--progress', '-p', action='store_true') ## Print progress during training

parser.add_argument('--name', type=str, default=None) ## Name of the file to save the experiment


args, unknown = parser.parse_known_args()

print('Starting experiment:')
print(f'Model = ResNet20')
print(f'Optimizer = SGD')
print(f'log10(lmbda)={np.log10(args.lmbda):.1f}, epochs={args.epochs}+10')
print(f'batch_size={args.batch_size}, log10(time_scale)={np.log10(args.time_scale):.1f}')


if args.name is not None:
    path = args.name + '.pkl.gz'
else:
    path = f'SGD_lmbda{np.log10(args.lmbda):.1f}_bs{args.batch_size}_ts{np.log10(args.time_scale):.1f}.pkl.gz'
os.makedirs("results", exist_ok=True)
path = os.path.join("results", path)

if os.path.exists(path):
    print('Experiments already exists, exiting')
    exit()


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./cifar10_data',
                                        train=True,
                                        download=True,
                                        transform=transform_train)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./cifar10_data',
                                       train=False,
                                       download=True,
                                       transform=transform_test)

test_loader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)

## Student model
resnet = ResNet20(in_channels=3, num_classes=10, VarProTraining=False)

## Learning problem

lmbda = args.lmbda
width = resnet.outer.in_features
lr = width * args.time_scale
print(f'Learning rate={lr:.3f}')

regularization = power_regularization(p=2)

criterion = TwoTimescaleCriterion(lmbda=lmbda, regularization_function=regularization, num_classes=10)

test_criterion = ClassifAccuracy(num_classes=10) # top_1 accuracy by default


## Training
# SGD: every parameter is trained with the same learning rate
optimizer = torch.optim.SGD(resnet.parameters(), lr=lr)

problem = LearningProblem(resnet,
                          train_loader,
                          optimizer,
                          criterion,
                          test_criterion=test_criterion,
                          test_loader=test_loader)

assert not hasattr(problem.criterion, 'projection')
assert problem.model.outer.weight.requires_grad
print('SGD Training!')

start = time.perf_counter()

problem.train_and_eval(args.epochs,
                       saving_step=1,
                       subprogress=args.progress,
                       averaging=False)

print('Changing learning rate for the last 10 epochs: lr=0.5*lr')

for param_group in problem.optimizer.param_groups:
    param_group['lr'] = 0.5 * lr

problem.train_and_eval(10,
                       saving_step=1,
                       subprogress=args.progress,
                       averaging=False)

stop = time.perf_counter()
elapsed_time = stop - start
print(f'Finished! Training took {elapsed_time:.0f} seconds')

## Saving dictionnary
dico = {
    'model': 'ResNet20',
    'optimizer': 'SGD',
    'model_state_list': problem.state_list,
    'loss_list': problem.loss_list,
    'accuracy_list': problem.test_loss_list,
    'epochs': args.epochs,
    'bacth_size': args.batch_size,
    'lmbda': lmbda,
    'time_scale': args.time_scale,
    'elapsed_time': elapsed_time
}

print('Saving dictionnary as: '+path)
with gzip.open(path, 'wb') as file:
    pickle.dump(dico, file)