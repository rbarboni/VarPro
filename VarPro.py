import torch
import torch.nn as nn
import numpy as np
import copy

from Utilities import *

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
    
class SignedSHLFeatureModel(nn.Module):
    def __init__(self, input_dim, width, activation, bias=False):
        super().__init__()
        self.weight = nn.Parameter(data=torch.randn(width, input_dim), requires_grad=True)
        self.sign = nn.Parameter(data=torch.randn(width), requires_grad=True)
        self.activation = activation
        if bias:
            self.bias = nn.Parameter(data=torch.zeros(width), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x):
        return self.activation(nn.functional.linear(x, self.weight, self.bias)) @ torch.diag(self.sign)
    
# feature model for a convolution
class ConvolutionFeatureModel(nn.Module):
    def __init__(self, input_dim, width, activation, scale):
        super().__init__()
        self.weight = nn.Parameter(data=torch.randn(width, input_dim), requires_grad=True)
        self.activation = activation
        self.scale = scale

    def forward(self, x):
        return self.activation(torch.linalg.vector_norm(self.weight.T[None,:,:] - x[:,:,None], dim=1) / self.scale)

## Models constructors
def SHL(input_dim, width, activation, bias=False, VarProTraining=True, clipper=None):
    feature_model = SHLFeatureModel(input_dim, width, activation, bias=bias)
    return VarProModel(feature_model, width, 1, VarProTraining=VarProTraining, clipper=clipper)

def SignedSHL(input_dim, width, activation, bias=False, VarProTraining=True, clipper=None):
    feature_model = SignedSHLFeatureModel(input_dim, width, activation, bias=bias)
    return VarProModel(feature_model, width, 1, VarProTraining=VarProTraining, clipper=clipper)

def Convolution(input_dim, width, activation, scale, VarProTraining=True, clipper=None):
    feature_model = ConvolutionFeatureModel(input_dim, width, activation, scale)
    return VarProModel(feature_model, width, 1, VarProTraining=VarProTraining, clipper=clipper)


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
        u = torch.linalg.solve(K + self.lmbda * torch.eye(width).to(K.device),  (features.T @ targets) / batch_size + self.lmbda)
        model.outer.weight = nn.Parameter(data=u.view((1,-1)), requires_grad=requires_grad)

# least square criterion with projection
class VarProCriterion(nn.Module):
    def __init__(self, lmbda):
        super().__init__()
        self.lmbda = lmbda
        self.projection = ExactRidgeProjection(lmbda=lmbda)
        
    def forward(self, inputs, targets, model):
        self.projection(inputs, targets, model)
        predictions = model(inputs)
        return 0.5 * ((predictions - targets)**2).mean() / self.lmbda  + 0.5 * (model.outer.weight**2).mean()

# least square criterion with projection and the unbiasing -1 term  
class VarProCriterionUnbiased(nn.Module):
    def __init__(self, lmbda):
        super().__init__()
        self.lmbda = lmbda
        self.projection = ExactRidgeProjectionUnbiased(lmbda=lmbda)
        
    def forward(self, inputs, targets, model):
        self.projection(inputs, targets, model)
        predictions = model(inputs)
        return 0.5 * ((predictions - targets)**2).mean() / self.lmbda  + 0.5 * ((model.outer.weight-1)**2).mean()
    
# least square criterion with general regularization
class TwoTimescaleCriterion(nn.Module):
    def __init__(self, lmbda, regularization_function):
        super().__init__()
        self.lmbda = lmbda
        self.regularization_function = regularization_function
        
    def forward(self, inputs, targets, model):
        predictions = model(inputs)
        return 0.5 * ((predictions - targets)**2).mean() / self.lmbda + self.regularization_function(model.outer.weight).mean()
    
## Regularization functions
class power_regularization():
    def __init__(self, p=2):
        self.p = p
        
    def __call__(self, x):
        return x**self.p / (self.p - 1)
    
class power_regularization_unbiased():
    def __init__(self, p=2):
        self.p = p
        
    def __call__(self, x):
        return (x-1)**self.p / (self.p - 1)

def  entropy_regularization(x):
    return x * torch.log(x) - x + 1


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
    
# thresholding weights value
class Thresholding():
    def __init__(self, x_min=-1, x_max=1):
        self.x_min = x_min
        self.x_max = x_max

    def __call__(self, w):
        return pmax(pmin(w, self.x_max), self.x_min)
    
class BallClipper():
    def __init__(self, radius=1):
        self.radius = radius

    def __call__(self, w):
        norm = w.norm(dim=1, keepdim=True).expand_as(w)
        return (w / norm) * pmin(norm, self.radius)

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

# Apply clipping function on feature model weights and signs (only for SignedSHL)
class FeatureSignClipper():
    def __init__(self, weight_clipper, sign_clipper):
        self.weight_clipper = weight_clipper
        self.sign_clipper = sign_clipper
        
    @torch.no_grad()
    def __call__(self, model):
        dico = model.state_dict()
        w, s = dico['feature_model.weight'], dico['feature_model.sign']
        w.copy_(self.weight_clipper(w))
        s.copy_(self.sign_clipper(s))
