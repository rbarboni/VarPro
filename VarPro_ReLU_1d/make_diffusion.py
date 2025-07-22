from Utilities import *
from scipy.integrate import solve_ivp
import gzip
import pickle
import os


import argparse

parser = argparse.ArgumentParser()

## Mandatory arguments
parser.add_argument('--gamma', type=float)
parser.add_argument('--epochs', type=int)

## Default arguments
  ## gamma controls the shape of the target distribution (converges to a dirac at gamma=\infty)
parser.add_argument('--grid_size', type=int, default=4096) ## Number of data samples
parser.add_argument('--time_scale', type=float, default=2**(-10)) ## Time scale of the gradient flow


parser.add_argument('--name', type=str, default=None) ## Name of the file to save the experiment

args, unknown = parser.parse_known_args()

print('Solving ultra-fast diffusion in 1d:')
print(f'gamma={args.gamma},')
print(f'grid_size={args.grid_size},  log2(time_scale)={np.log2(args.time_scale):.1f}, time_steps={args.time_steps}')

if args.name is not None:
    path = args.name + '.pkl.gz'
else:
    path = f'diffusion_relu1d_gamma{args.gamma:.0f}_ts{np.log2(args.time_scale):.0f}.pkl.gz'

if os.path.exists(path):
    print('File already exists, exiting')
    exit()

M = args.grid_size
gamma = args.gamma

X = np.linspace(-np.pi, np.pi, M+1)
X = 0.5 * (X[1:]+X[:-1])

f0 = normalize(np.ones(M)) ## initialization
f_star = normalize((1 / (1 + gamma * np.sin(X/2)**2)) + (0.5 / (1 + 1 * gamma * np.sin((X-0.4*np.pi)/2)**2))) ## target

def F(t, f):
    return - 0.5 * f * Laplacian((f_star / f)**2, h=2*np.pi/M) - 0.5 * Grad((f_star / f)**2, h=2*np.pi/M) * Grad(f, h=2*np.pi/M)

T = args.epochs
time_scale = args.time_scale

t_eval = np.linspace(0, T*time_scale, T+1)
t_span = (0, T*time_scale)

print('Call to solve_ivp')
sol = solve_ivp(F, t_span, f0, t_eval=t_eval, method='LSODA')

print('Saving file at: '+path)
print('(can take several minutes)')
with gzip.open(path, 'wb') as file:
    pickle.dump(sol.y.T, file)