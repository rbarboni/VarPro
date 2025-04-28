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
parser.add_argument('--epochs', type=int) ## Number of epochs
parser.add_argument('--lmbda', type=float)  ## Regularization parameter

## Default arguments
parser.add_argument('--batch_size', '-bs', type=int, default=128) ## Number of data samples
parser.add_argument('--time_scale', '-ts', type=float, default=2**(-10)) ## Time scale of the gradient flow
parser.add_argument('--seed', type=int, default=0)  ## Random seed
parser.add_argument('--progress', type=bool, default=False) ## Print progress during training
parser.add_argument('--saving_step', type=int, default=1) ## Save the model every saving_step epochs

parser.add_argument('--name', type=str, default=None) ## Name of the file to save the experiment


args, unknown = parser.parse_known_args()

print('Starting experiment:')
print(f'log10(lmbda)={np.log10(args.lmbda):.1f}, epochs={args.epochs}')
print(f'batch_size={args.batch_size}, log2(time_scale)={np.log2(args.time_scale):.1f}, seed={args.seed}')


if args.name is not None:
    path = 'results/' + args.name + '.pkl.gz'
else:
    path = f'lmbda{np.log10(args.lmbda):.1f}_bs{args.batch_size}_ts{np.log2(args.time_scale):.1f}_seed{args.seed}.pkl.gz'

if os.path.exists(path):
    print('Experiments already exists, exiting')
    exit()

torch.manual_seed(args.seed)


## Dataset
transform_train = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32)])

trainset = torchvision.datasets.MNIST(root='./mnist_data',
                                    train=True,
                                    download=True,
                                    transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)

testset = torchvision.datasets.MNIST(root='./mnist_data',
                                    train=False,
                                    download=True,
                                    transform=transform_train)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=1)

## Student model
resnet = ResNet18(in_channels=1, VarProTraining=True)

## Learning problem

lmbda = args.lmbda
lr = 512 * args.time_scale

criterion = VarProCriterion(lmbda=lmbda, num_classes=10)
test_criterion = VarProClassifEvalutation(lmbda=lmbda, num_classes=10)

#print('Performing 1 projection step before training')
#inputs, targets = next(iter(train_loader))
#inputs, targets = inputs.to(device), targets.to(device)
#student.to(device)
#criterion.projection(inputs, targets, resnet)
#student.to(torch.device('cpu'))

optimizer = torch.optim.SGD(resnet.feature_model.parameters(), lr=lr)
problem = LearningProblem(resnet, train_loader, optimizer, criterion, test_criterion=test_criterion, test_loader=test_loader)

if __name__ == '__main__':
    ## Training
    # VarPro training: only the feature model is trained
    assert hasattr(problem.criterion, 'projection')
    assert not problem.model.outer.weight.requires_grad
    print('VarPro Training!')

    start = time.perf_counter()
    problem.train_and_eval(args.epochs, saving_step=args.saving_step, subprogress=args.progress)
    stop = time.perf_counter()
    elapsed_time = stop - start
    print(f'Finished! Training took {elapsed_time:.0f} seconds')


    ## Saving Problem
    #print('Saving experiment as: '+path)
    #with gzip.open(path, 'wb') as file:
    #    pickle.dump(problem, file)

    ## Saving dictionnary
    dico = {
        'model_state_list': problem.state_list,
        'loss_list': problem.loss_list,
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