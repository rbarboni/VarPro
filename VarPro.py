import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import ot
import copy

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using gpu !')
else:
    device = torch.device('cpu')
    print('WARNING: using cpu, computation may be slow !')

## Dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels):
        super().__init__()
        self.inputs = inputs
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

## Training routines
def training_loop(model, train_loader, optimizer, criterion):
    loss_list = []
    model.to(device)
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(inputs, targets, model)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        if hasattr(model, 'clipper'):
            model.clipper(model)
    model.to(torch.device('cpu')) 
    return loss_list

class LearningProblem():
    def __init__(self, model, train_loader, optimizer, criterion):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.loss_list = []
        self.state_list = [copy.deepcopy(self.model.state_dict())]

    def train(self, epochs, progress=True, saving_step=1):
        iterator = tqdm(range(epochs)) if progress else range(epochs)
        for i in iterator:
            loss = training_loop(self.model, self.train_loader, self.optimizer, self.criterion)
            self.loss_list.extend(loss)
            if (i+1) % saving_step == 0:
                self.state_list.append(copy.deepcopy(self.model.state_dict()))
            if progress:
                iterator.set_description(f'log10(loss) = {np.log10(self.loss_list[-1]):.2f}')
            elif (i+1) % (epochs // 100) == 0:
                print(f'{100 * (i+1) / epochs:.0f}% elapsed, log10(loss)={np.log10(self.loss_list[-1]):.2f}')

## Models
class ActivationFunction(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f
        
    def forward(self, x):
        return self.f(x)

# VarPro model are the composition of a feature model and an outer layer
class VarProModel(nn.Module):
    def __init__(self, feature_model, width, output_dim, VarProTraining=True, clipper=None):
        super().__init__()
        self.feature_model = feature_model
        self.VarProTraining = VarProTraining
        self.width = width
        self.outer = nn.Linear(width, output_dim, bias=False)
        if VarProTraining:
            self.outer.weight.requires_grad = False
        if clipper is not None:
            self.clipper = clipper
            self.clipper(self)

    def forward(self, inputs):
        return self.outer(self.feature_model(inputs)) / self.width ## normalized output function

# feature model of a SHL
class SHLFeatureModel(nn.Module):
    def __init__(self, input_dim, width, activation, bias=False):
        super().__init__()
        self.weight = nn.Parameter(data=torch.randn(width, input_dim), requires_grad=True)
        self.activation = activation
        self.bias = nn.Parameter(data=torch.zeros(width), requires_grad=bias)

    def forward(self, x):
        return self.activation(nn.functional.linear(x, self.weight, self.bias))
    
# feature model for a convolution
class ConvolutionFeatureModel(nn.Module):
    def __init__(self, input_dim, width, activation, scale):
        super().__init__()
        self.weight = nn.Parameter(data=torch.randn(width, input_dim), requires_grad=True)
        self.activation = activation
        self.scale = scale

    def forward(self, x):
        return self.activation(torch.linalg.vector_norm(self.weight.T[None,:,:] - x[:,:,None], dim=1) / self.scale)

def SHL(input_dim, width, activation, bias=False, VarProTraining=True, clipper=None):
    feature_model = SHLFeatureModel(input_dim, width, activation, bias=bias)
    return VarProModel(feature_model, width, 1, VarProTraining=VarProTraining, clipper=clipper)

def Convolution(input_dim, width, activation, scale, VarProTraining=True, clipper=None):
    feature_model = ConvolutionFeatureModel(input_dim, width, activation, scale)
    return VarProModel(feature_model, width, 1, VarProTraining=VarProTraining, clipper=clipper)


def freeze(module):
    for p in module.parameters():
        p.requires_grad = False

## Criterions
# perform projection on the outer layer
class ExactRidgeProjection():
    def __init__(self, lmbda):
        self.lmbda = lmbda

    @torch.no_grad()
    def __call__(self, inputs, targets, model, requires_grad=False):
        features = model.feature_model(inputs).clone().detach()
        batch_size, width = features.shape[0], features.shape[1]
        if batch_size > width: ## underparameterized case
            K = (features.T @ features) / (batch_size * width)
            u = torch.linalg.solve(K + self.lmbda * torch.eye(width).to(K.device),  (features.T @ targets) / batch_size)
        else: ## overparameterized case
            K = (features @ features.T) / (batch_size*width)
            u = features.T @ torch.linalg.solve(K + self.lmbda * torch.eye(batch_size).to(K.device), targets) / batch_size
        model.outer.weight = nn.Parameter(data=u.view((1,-1)), requires_grad=requires_grad)

# perform projection on the outer layer with the unbiasing -1 term
class ExactRidgeProjectionUnbiased():
    def __init__(self, lmbda):
        self.lmbda = lmbda
    
    @torch.no_grad()
    def __call__(self, inputs, targets, model, requires_grad=False):
        features = model.feature_model(inputs).clone().detach()
        batch_size, width = features.shape[0], features.shape[1]
        K = (features.T @ features) / (batch_size * width)
        u = torch.linalg.solve(K + self.lmbda * torch.eye(width),  (features.T @ targets) / batch_size + self.lmbda)
        model.outer.weight = nn.Parameter(data=u.view((1,-1)), requires_grad=requires_grad)

# least square criterion with projection
class LeastSquareCriterion(nn.Module):
    def __init__(self, lmbda, projection=None):
        super().__init__()
        self.lmbda = lmbda
        self.projection = projection
        
    def forward(self, inputs, targets, model):
        if self.projection is not None:
            self.projection(inputs, targets, model)
        predictions = model(inputs)
        return 0.5 * ((predictions - targets)**2).mean() / self.lmbda  + 0.5 * (model.outer.weight**2).mean()

# least square criterion with projection and the unbiasing -1 term  
class LeastSquareCriterionUnbiased(nn.Module):
    def __init__(self, lmbda, projection=None):
        super().__init__()
        self.lmbda = lmbda
        self.projection = projection
        
    def forward(self, inputs, targets, model):
        if self.projection is not None:
            self.projection(inputs, targets, model)
        predictions = model(inputs)
        return 0.5 * ((predictions - targets)**2).mean() / self.lmbda  + 0.5 * ((model.outer.weight-1)**2).mean()

## Clipping
# clipping function for the features 
class Normalization():
    def __call__(self, w):
        return w / torch.norm(w, 2, dim=-1, keepdim=True).expand_as(w)    

class PeriodicBoundaryCondition():
    def __init__(self, x_min=-1, x_max=1):
        self.x_min = x_min
        self.x_max = x_max

    def __call__(self, w):
        return self.x_min + (w-self.x_min) % (self.x_max-self.x_min)

# apply a clipping function to the features
class FeatureClipper():
    def __init__(self, clipping_function):
        self.clipping_function = clipping_function
        
    @torch.no_grad()
    def __call__(self, model):
        w = model.state_dict()['feature_model.weight']
        w.copy_(self.clipping_function(w))

# apply a clipping function to the features and the bias
class FeatureBiasClipper():
    def __init__(self, weight_clipper, bias_clipper):
        self.weight_clipper = weight_clipper
        self.bias_clipper = bias_clipper
        
    @torch.no_grad()
    def __call__(self, model):
        dico = model.state_dict()
        w, b = dico['feature_model.weight'], dico['feature_model.bias']
        w.copy_(self.weight_clipper(w))
        b.copy_(self.bias_clipper(b))

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

def compute_distance(distance, weight_list, weight_ref, c_ref=None, c_list=None, N_eval=100, progress=True):
    # uniform coefficients
    if c_ref is None:
        c_ref = torch.ones(weight_ref.shape[0]) / weight_ref.shape[0]
    if c_list is None:
        c = torch.ones(weight_list[0].shape[0]) / weight_list[0].shape[0]
        c_list = [c for _ in weight_list]
    
    distance_list = []
    idx = np.array([int(i) for i in np.linspace(0, len(weight_list)-1, N_eval+1)])
    iterator = tqdm(idx) if progress else idx
    for i in iterator:
        distance_list.append(distance(weight_ref, c_ref, weight_list[i], c_list[i]).item())
    return distance_list, idx

## Utilities
def add_one_row(x):
    return torch.cat((x, torch.ones(x.shape[0]).view((x.shape[0],1))), dim=1)

def model_plot_2d(model, add_one=False, N_points=100, x_lim=(-5,5)): ## Data plot
    x = np.linspace(*x_lim, N_points)
    y = np.linspace(*x_lim, N_points)
    xx, yy = np.meshgrid(x, y)
    if add_one:
        inputs = torch.tensor(np.stack((xx.flatten(), yy.flatten(), np.ones(N_points**2))), dtype=torch.float32).T
    else:
        inputs = torch.tensor(np.stack((xx.flatten(), yy.flatten())), dtype=torch.float32).T
    zz = model(inputs).detach().view((N_points, N_points)).numpy()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx, yy, zz, linewidth=0, antialiased=False, cmap=cm.coolwarm)
    fig.colorbar(surf)
    plt.show()

def model_plot_1d(model, add_one=False, N_points=1000, x_lim=(-10,10)): ## Data plot
    x = np.linspace(*x_lim, N_points)
    if add_one:
        inputs = torch.tensor(np.stack((x, np.ones(N_points))), dtype=torch.float32).T
    else:
        inputs = torch.tensor(x, dtype=torch.float32).view((N_points, -1))
    y = model(inputs).detach().squeeze().numpy()
    plt.plot(x, y)
    plt.show()

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