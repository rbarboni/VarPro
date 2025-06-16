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

## Fix random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

activation = nn.ReLU() ## activation function
clipper = FeatureClipper(Normalization())


## Teacher model
teacher_width = args.teacher_width
teacher = SHL(2, teacher_width, activation, bias=False, clipper=clipper)

## Teacher feature distribution
# the teacher distribution approximates a dirac, the parameter gamma constrols the shape of the distribution
gamma = args.gamma

modes = np.pi * np.array([0, 0.4])

Theta = np.pi * generate_periodic_distribution(teacher_width, dim=1, gamma=gamma).squeeze()

Theta[1::3] += modes[1]

teacher_init = torch.tensor([[np.cos(theta), np.sin(theta)] for theta in Theta], dtype=torch.float32)

teacher.feature_model.weight = nn.Parameter(data=teacher_init) ## teacher feature distribution
teacher.outer.weight = nn.Parameter(data=torch.ones_like(teacher.outer.weight), requires_grad=False) ## teacher outer weight

teacher.clipper(teacher)
teacher.apply(freeze)


## Dataset
inputs = torch.randn(args.N,2)
targets = teacher(inputs)
dataset = CustomDataset(inputs, targets)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))

## Student model
student_width = args.student_width

student = SHL(2, student_width, activation, bias=False, clipper=clipper, VarProTraining=True)

student_init = torch.randn((student_width, 2), dtype=torch.float32)
student.feature_model.weight = nn.Parameter(data=student_init, requires_grad=True)
student.clipper(student)

## Learning problem

lmbda = args.lmbda
lr = student_width * args.time_scale

criterion = VarProCriterion(lmbda=lmbda)

print('Performing 1 projection step before training')
inputs, targets = next(iter(train_loader))
criterion.projection(inputs, targets, student)

optimizer = torch.optim.SGD([student.feature_model.weight], lr=lr)
problem = LearningProblem(student, train_loader, optimizer, criterion)

## Training
# VarPro training: only the feature model is trained
assert hasattr(problem.criterion, 'projection')
assert not problem.model.outer.weight.requires_grad
print('VarPro Traning!')

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


## Exact solution in 1d
print('Computing MMD distance to exact diffusion')

with gzip.open(f'../diffusion_relu1d_gamma{args.gamma:.0f}_ts-10.pkl.gz', 'rb') as file:
    f_list = pickle.load(file)

T_diffusion = f_list.shape[0] - 1
M = f_list.shape[1]
X = np.linspace(-np.pi, np.pi, M+1)
X = 0.5 * (X[1:]+X[:-1])

w2 = torch.tensor([[np.cos(x), np.sin(x)] for x in X], dtype=torch.float32)

#ts_ratio = args.time_scale / 2**(-8)
T_max = min(args.epochs, T_diffusion)

distance_diffusion_list = []
distance_diffusion_idx = [int(i) for i in np.linspace(0, T_max, min(T_max+1, 1001))]

for i in distance_diffusion_idx:
    w1 = problem.state_list[i]['feature_model.weight']
    c2 = torch.tensor(f_list[i],dtype=torch.float32) * 2*np.pi / M
    distance_diffusion_list.append(compute_distance(DistanceMMD(), w1, w2, c2=c2).item())


## Saving Problem
#print('Saving experiment as: '+path)
#with gzip.open(path, 'wb') as file:
#    pickle.dump(problem, file)

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
    'distance_diffusion_list': distance_diffusion_list,
    'distance_diffusion_idx': distance_diffusion_idx,
    'elapsed_time': elapsed_time
}

print('Saving dictionnary as: '+path)
with gzip.open(path, 'wb') as file:
    pickle.dump(dico, file)