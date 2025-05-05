import sys
sys.path.append('..')

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

## Default arguments
parser.add_argument('--gamma', type=float, default=100)  ## gamma controls the shape of the target distribution (converges to a dirac at gamma=\infty)
parser.add_argument('--N', '-N', type=int, default=4096) ## Number of data samples
parser.add_argument('--teacher_width', type=int, default=4096) ## Width of the teacher model
parser.add_argument('--time_scale', type=float, default=2**(-10)) ## Time scale of the gradient flow
parser.add_argument('--seed', type=int, default=0)  ## Random seed
parser.add_argument('--progress', type=bool, default=False) ## Print progress during training
parser.add_argument('--saving_step', type=int, default=1) ## Save the model every saving_step epochs

parser.add_argument('--name', type=str, default=None) ## Name of the file to save the experiment


args, unknown = parser.parse_known_args()

print('Starting experiment:')
print(f'student_width={args.student_width}, log10(lmbda)={np.log10(args.lmbda):.1f}, epochs={args.epochs}')
print(f'N={args.N}, gamma={args.gamma}, log2(time_scale)={np.log2(args.time_scale):.1f}, seed={args.seed}')

if args.name is not None:
    path = args.name + '.pkl.gz'
else:
    path = f'width{args.student_width}_lmbda{np.log10(args.lmbda):.1f}_gamma{args.gamma:.1f}_N{args.N}_ts{np.log2(args.time_scale):.1f}_seed{args.seed}.pkl.gz'

if os.path.exists(path):
    print('Experiments already exists, exiting')
    exit()

torch.manual_seed(args.seed)



scale = 0.5
activation = ActivationFunction(lambda t: 2*torch.exp(-t/scale) / scale**2) ## activation function

w_lim = 2
clipping_function = PeriodicBoundaryCondition(x_min=-w_lim, x_max=w_lim)
clipper = FeatureClipper(clipping_function)


## Teacher model
teacher_width = args.teacher_width

modes = w_lim * torch.tensor([[-0.5, 0], [0.5, 0.5]], dtype=torch.float32)

gamma = args.gamma

teacher_weight = torch.tensor(generate_periodic_distribution(teacher_width, dim=2, gamma=gamma), dtype=torch.float32)
teacher_weight[::2] += modes[0]
teacher_weight[1::2] += modes[1]


teacher = Convolution(2, teacher_width, activation, clipper=clipper)

teacher.feature_model.weight = nn.Parameter(data=teacher_weight) ## teacher feature distribution
teacher.outer.weight = nn.Parameter(data=torch.ones_like(teacher.outer.weight, dtype=torch.float32)) ## teacher outer weight

teacher.clipper(teacher)
teacher.apply(freeze)


## Dataset
inputs = torch.randn(args.N,2)
targets = teacher(inputs)
dataset = CustomDataset(inputs, targets)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))

## Student model
student_width = args.student_width

student = Convolution(2, student_width, activation, clipper=clipper, VarProTraining=True)

student_init = w_lim * (2 * torch.rand((student_width, 2), dtype=torch.float32) - 1)
student.feature_model.weight = nn.Parameter(data=student_init, requires_grad=True)
student.clipper(student)

## Learning problem

lmbda = args.lmbda
lr = student_width * args.time_scale

criterion = VarProCriterionUnbiased(lmbda=lmbda)

print('Performing 1 projection step before training')
inputs, targets = next(iter(train_loader))
#inputs, targets = inputs.to(device), targets.to(device)
#student.to(device)
criterion.projection(inputs, targets, student)
#student.to(torch.device('cpu'))

optimizer = torch.optim.SGD([student.feature_model.weight], lr=lr)
problem = LearningProblem(student, train_loader, optimizer, criterion)

## Training
# VarPro training: only the feature model is trained
assert hasattr(problem.criterion, 'projection')
assert not problem.model.outer.weight.requires_grad
print('VarPro Training!')

start = time.perf_counter()
problem.train(args.epochs, progress=args.progress, saving_step=args.saving_step)
stop = time.perf_counter()
elapsed_time = stop - start
print(f'Finished! Training took {elapsed_time:.0f} seconds')


## MMD distance to teacher
print('Computing MMD distance to teacher')

distance_teacher_list = []
distance_teacher_idx = [int(i) for i in np.linspace(0, args.epochs, 1001)]
for i in distance_teacher_idx:
    w1 = problem.state_list[i]['feature_model.weight']
    w2 = teacher.feature_model.weight
    distance_teacher_list.append(compute_distance(DistanceMMD(), w1, w2).item())

## Saving dictionnary
dico = {
    'student_state_list': problem.state_list,
    'teacher_state': copy.deepcopy(teacher.state_dict()),
    'loss_list': problem.loss_list,
    'student_width': student_width,
    'teacher_width': teacher_width,
    'gamma': gamma,
    'epochs': args.epochs,
    'N': args.N,
    'seed': args.seed,
    'lmbda': lmbda,
    'time_scale': args.time_scale,
    'distance_teacher_list': distance_teacher_list,
    'distance_teacher_idx': distance_teacher_idx,
    'elapsed_time': elapsed_time
}

print('Saving dictionnary as: '+path)
with gzip.open(path, 'wb') as file:
    pickle.dump(dico, file)