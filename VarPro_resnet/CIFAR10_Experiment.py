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
parser.add_argument('--lmbda', '-l', type=float)  ## Regularization parameter

## Default arguments
parser.add_argument('--batch_size', '-bs', type=int, default=128) ## Number of data samples
parser.add_argument('--time_scale', '-ts', type=float, default=1e-4) ## Time scale of the gradient flow
parser.add_argument('--seed', '-s', type=int, default=0)  ## Random seed
parser.add_argument('--progress', '-p', type=bool, default=False) ## Print progress during training
parser.add_argument('--model', '-m', type=str, default='ResNet18') ## Model to use

parser.add_argument('--name', type=str, default=None) ## Name of the file to save the experiment


args, unknown = parser.parse_known_args()

print('Starting experiment:')
print(f'log10(lmbda)={np.log10(args.lmbda):.1f}, epochs={args.epochs}+3')
print(f'batch_size={args.batch_size}, log10(time_scale)={np.log10(args.time_scale):.1f}, seed={args.seed}')


if args.name is not None:
    path = args.name + '.pkl.gz'
else:
    path = 'CIFAR10_'+args.model+f'_lmbda{np.log10(args.lmbda):.1f}_bs{args.batch_size}_ts{np.log10(args.time_scale):.1f}_seed{args.seed}.pkl.gz'

if os.path.exists(path):
    print('Experiments already exists, exiting')
    exit()

torch.manual_seed(args.seed)


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
model_dict = {'ResNet9': ResNet9,
              'ResNet18': ResNet18,
              'SimpleResNet9': SimpleResNet9,
              'SimpleResNet18': SimpleResNet18,}

resnet = model_dict[args.model](in_channels=3, num_classes=10, VarProTraining=True)

## Learning problem

lmbda = args.lmbda
lr = 512 * args.time_scale

criterion = VarProCriterion(lmbda=lmbda, num_classes=10)

test_criterion = ClassifAccuracy(num_classes=10) # top_1 accuracy by default

optimizer = torch.optim.SGD(resnet.feature_model.parameters(), lr=lr)

problem = LearningProblem(resnet,
                          train_loader,
                          optimizer,
                          criterion,
                          test_criterion=test_criterion,
                          test_loader=test_loader)

## Training
# VarPro training: only the feature model is trained
assert hasattr(problem.criterion, 'projection')
assert not problem.model.outer.weight.requires_grad
print('VarPro Training!')

start = time.perf_counter()

problem.train_and_eval(args.epochs,
                       saving_step=1,
                       subprogress=args.progress,
                       averaging=True)

print('Changing learning rate for the last 3 epochs: lr=0.1*lr')

for param_group in problem.optimizer.param_groups:
    param_group['lr'] = 0.1 * lr

problem.train_and_eval(3,
                       saving_step=1,
                       subprogress=args.progress,
                       averaging=True)

stop = time.perf_counter()
elapsed_time = stop - start
print(f'Finished! Training took {elapsed_time:.0f} seconds')

## Saving dictionnary
dico = {
    'model_state_list': problem.state_list,
    'loss_list': problem.loss_list,
    'accuracy_list': problem.test_loss_list,
    'epochs': args.epochs,
    'bacth_size': args.batch_size,
    'seed': args.seed,
    'lmbda': lmbda,
    'time_scale': args.time_scale,
    'elapsed_time': elapsed_time
}

print('Saving dictionnary as: '+path)
with gzip.open(path, 'wb') as file:
    pickle.dump(dico, file)