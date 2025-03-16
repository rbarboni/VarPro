from VarPro import *

import argparse
import time
import gzip
import pickle
import os

parser = argparse.ArgumentParser()

## Mandatory arguments
parser.add_argument('--student_width', type=int) ## Width of the student model
parser.add_argument('--epochs', type=int) ## Number of epochs
parser.add_argument('--lmbda', type=float)  ## Regularization parameter
parser.add_argument('--gamma', type=float)  ## gamma controls the shape of the target distribution (converges to a dirac at gamma=\infty)

## Default arguments
parser.add_argument('--N', '-N', type=int, default=10000) ## Number of data samples
parser.add_argument('--time_scale', type=float, default=2**(-11)) ## Time scale of the gradient flow
parser.add_argument('--seed', type=int, default=0)  ## Random seed
parser.add_argument('--progress', type=bool, default=False) ## Print progress during training
parser.add_argument('--varpro', type=bool, default=True) ## Use VarPro training or not (classical SHL training)
parser.add_argument('--saving_step', type=int, default=10) ## Save the model every saving_step epochs

parser.add_argument('--name', type=str, default=None) ## Name of the file to save the experiment


args, unknown = parser.parse_known_args()

print('Starting experiment:')
print(f'student_width={args.student_width}, log10(lmbda)={np.log10(args.lmbda):.1f}, epochs={args.epochs}')
print(f'N={args.N}, gamma={args.gamma}, log2(time_scale)={np.log2(args.time_scale):.1f}, seed={args.seed}')

torch.manual_seed(args.seed)

## Data

# Teacher function is calculated in closed form instead of using a neural network
# corresponds to teacher with feature distribution p(Theta) \propro 1 / (1 + gamma_star * np.sin(Theta)**2) (2 diracs)
def teacher_model(inputs, gamma):
    rotated_inputs = inputs @ torch.tensor([[1, 0], [0, (1+gamma)**0.5]])
    norm = rotated_inputs.norm(dim=1)
    cos = rotated_inputs[:,0] / norm
    sin = rotated_inputs[:,1] / norm
    return norm * (cos * torch.asinh(cos*gamma**0.5) + sin * torch.asin(sin*(gamma/(1+gamma))**0.5)) / (np.pi*gamma**0.5)

inputs = torch.randn(args.N,2)
targets = teacher_model(inputs, args.gamma).view((args.N, 1))
dataset = CustomDataset(inputs, targets)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))

## Student model

activation = nn.ReLU() ## activation function
clipper = FeatureClipper(Normalization())

student_width = args.student_width
student_init = torch.randn((student_width, 2), dtype=torch.float32)

student = SHL(2, student_width, activation, bias=False, clipper=clipper, VarProTraining=args.varpro)

student.feature_model.weight = nn.Parameter(data=student_init, requires_grad=True)
student.clipper(student)

## Learning problem

lmbda = args.lmbda
lr = student_width * args.time_scale

projection = ExactRidgeProjection(lmbda=lmbda)

print('Performing 1 projection step before training')
inputs, targets = next(iter(train_loader))
inputs, targets = inputs.to(device), targets.to(device)
student.to(device)
projection(inputs, targets, student, requires_grad=(not args.varpro))
student.to(torch.device('cpu'))

if args.varpro:
    criterion = LeastSquareCriterion(lmbda=lmbda, projection=projection)
else:
    criterion = LeastSquareCriterion(lmbda=lmbda, projection=None)

optimizer = torch.optim.SGD([student.feature_model.weight], lr=lr)
if not args.varpro:
    optimizer.add_param_group({'lr': 5 * lmbda * student_width, 'params': student.outer.weight})

problem = LearningProblem(student, train_loader, optimizer, criterion)

## Training
# VarPro training: only the feature model is trained
# SHL training: the feature model is trained and the outer weights are trained
if args.varpro:
    assert problem.criterion.projection is not None
    assert not problem.model.outer.weight.requires_grad
    print('VarPro Traning!')
else:
    assert problem.criterion.projection is None
    assert problem.model.outer.weight.requires_grad
    print('SHL Training!')

start = time.perf_counter()
saving_step = 10
problem.train(args.epochs, progress=args.progress, saving_step=args.saving_step)
stop = time.perf_counter()
print(f'Finished! Training took {stop-start:.0f} seconds')


## Saving Problem

print('Saving files')
if args.name is not None:
    path = args.name
else:
    path = f'width{student_width}_lmbda{np.log10(lmbda):.1f}_ts{np.log2(args.time_scale):.1f}_gamma{args.gamma:.1f}_seed{args.seed}_N{args.N}'
    if args.varpro:
        path = 'VarPro_'+path
    else:
        path = 'SHL_'+path
path = path + '.pkl.gz'

folder = 'SHL_ReLU1d_experiments/'
if not os.path.exists(folder):
    os.makedirs(folder)
    
complete_path = os.path.join(folder, path)
with gzip.open(complete_path, 'wb') as file:
    pickle.dump(problem, file)

print('Experiment saved as: '+complete_path)
