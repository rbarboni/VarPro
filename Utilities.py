import torch
import torch.nn as nn
import numpy as np
import ot
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from tqdm import tqdm



## Utilities
def freeze(module):
    for p in module.parameters():
        p.requires_grad = False

def add_one_row(x):
    return torch.cat((x, torch.ones(x.shape[0]).view((x.shape[0],1))), dim=1)

def model_plot_1d(model, add_one=False, N_points=1000, x_lim=(-10,10), plot=True): ## Data plot
    x = np.linspace(*x_lim, N_points)
    if add_one:
        inputs = torch.tensor(np.stack((x, np.ones(N_points))), dtype=torch.float32).T
    else:
        inputs = torch.tensor(x, dtype=torch.float32).view((N_points, -1))
    y = model(inputs).detach().squeeze().numpy()
    if plot:
        plt.plot(x, y)
        plt.show()
    return x, y

def model_plot_2d(model, add_one=False, N_points=100, x_lim=(-5,5), plot=True): ## Data plot
    x = np.linspace(*x_lim, N_points)
    y = np.linspace(*x_lim, N_points)
    xx, yy = np.meshgrid(x, y)
    if add_one:
        inputs = torch.tensor(np.stack((xx.flatten(), yy.flatten(), np.ones(N_points**2))), dtype=torch.float32).T
    else:
        inputs = torch.tensor(np.stack((xx.flatten(), yy.flatten())), dtype=torch.float32).T
    zz = model(inputs).detach().view((N_points, N_points)).numpy()
    if plot:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(xx, yy, zz, linewidth=0, antialiased=False, cmap=cm.coolwarm)
        fig.colorbar(surf)
        plt.show()
    return xx, yy, zz

def weight_animation_2d(weight_array, name='animation.mp4'):
    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot()
    
    x_min, x_max = np.min(weight_array[:,:,0])-0.2, np.max(weight_array[:,:,0])+0.2
    y_min, y_max = np.min(weight_array[:,:,1])-0.2, np.max(weight_array[:,:,1])+0.2
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    
    points, = ax.plot([], [], 'o', c='blue', ms=3)
    def animate(i):
        points.set_data(weight_array[i,:,0], weight_array[i,:,1])
        return points
    
    time = 20
    fps = 10
    frames = time * fps
    rate = weight_array.shape[0] // frames
    anim = animation.FuncAnimation(fig, lambda i: animate(rate*i), frames, blit=False, interval=1)
    writer = animation.FFMpegWriter(fps=fps)
    anim.save(name, writer = writer)

def generate_periodic_distribution(N, dim=2, scale=1, gamma=100):
    theta_list = []
    for _ in range(dim):
        Theta = 2 * np.pi * np.random.rand(N) - np.pi
        W = torch.tensor([[np.cos(theta), np.sin(theta)] for theta in Theta], dtype=torch.float32)
        W = W @ torch.tensor([[(1+gamma)**0.5, 0], [0, 1]], dtype=torch.float32)
        W = W / torch.norm(W, 2, dim=-1, keepdim=True).expand_as(W)
        Theta = circle_to_line(W.numpy())
        Theta = (2*Theta + np.pi) % (2*np.pi) - np.pi
        Theta = Theta / np.pi
        theta_list.append(Theta)
    return np.stack(theta_list, axis=1)

## density of diracs (with position in x on given interval) convolved with gaussians
def gaussian_conv(x, coef=None, scale=1, interval=(-np.pi, np.pi), N_points=1000):
    res = np.zeros(N_points)
    z_min, z_max = interval
    z = np.linspace(z_min, z_max, N_points+1)
    z = 0.5*(z[1:]+z[:-1])
    if coef is None:
        coef = np.ones(len(x)) / len(x)
    for i in range(len(x)):
        res += coef[i] * np.exp(- 0.5 * (z_min + (z-x[i]-z_min) % (z_max-z_min))**2 / scale**2)
    return z, normalize(res, interval=interval)

def circle_to_line(x):
    return 2 * np.arctan( x[:,1] / (1+x[:,0]))

def Laplacian(f, h=1):
    return (np.roll(f, 1) + np.roll(f, -1) - 2*f) / h**2

def Grad(f, h=1):
    return (f - np.roll(f, 1)) / h

def center(f):
    return f - f.mean()

def normalize(f, interval=(-np.pi,np.pi)):
    z_min, z_max = interval
    return len(f) * f / (f.sum() * (z_max-z_min))

def pmax(x, t):
    return torch.maximum(x, t*torch.ones_like(x))

def pmin(x, t):
    return torch.minimum(x, t*torch.ones_like(x))

## Distance functions
class GaussianKernel():
    def __init__(self, gamma=1):
        self.gamma = gamma
        
    def __call__(self, x):
        return torch.exp(- 0.5 * x.norm(dim=-1)**2 / self.gamma**2)

class EnergyKernel():
    def __call__(self, x):
        return -x.norm(dim=-1)

class DistanceMMD():
    def __init__(self, kernel=EnergyKernel(), projection=nn.Identity()):
        self.kernel = kernel
        self.projection = projection

    def __call__(self, m1, c1, m2, c2):
        K1 = self.kernel(self.projection(m1[:,None,:] - m1[None,:,:]))
        K2 = self.kernel(self.projection(m2[:,None,:] - m2[None,:,:]))
        K3 = self.kernel(self.projection(m1[:,None,:] - m2[None,:,:]))
        return ( c1.dot(K1 @ c1) + c2.dot(K2 @ c2) - 2 * c1.dot(K3 @ c2) ).sqrt()

class DistanceOT():
    def __init__(self, projection=nn.Identity()):
        self.projection = projection

    def __call__(self, m1, c1, m2, c2):
        M = (self.projection(m1[:,None,:] - m2[None,:,:])**2).sum(dim=-1).numpy()
        return np.sqrt(ot.emd2(c1.numpy(), c2.numpy(), M))

def compute_distance(distance, w1, w2, c1=None, c2=None):
    # uniform coefficients
    if c1 is None:
        c1 = torch.ones(w1.shape[0]) / w1.shape[0]
    if c2 is None:
        c2 = torch.ones(w2.shape[0]) / w2.shape[0]
    
    return distance(w1, c1, w2, c2)