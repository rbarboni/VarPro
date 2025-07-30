import torch
import torch.nn as nn
import torchmetrics
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
def training_loop(model, train_loader, optimizer, criterion, progress=False):
    loss_list = []
    model.to(device)
    model.train()

    iterator = tqdm(train_loader) if progress else train_loader
    for inputs, targets in iterator:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(inputs, targets, model)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        if hasattr(model, 'clipper'):
            model.clipper(model)
        if progress:
            iterator.set_description(f'log10(loss) = {np.log10(loss_list[-1]):.2f}')
    
    model.to(torch.device('cpu'))
    return loss_list

def evaluation_loop(model, loader, criterion):
    loss_list = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            loss = criterion(inputs, targets, model)
            loss_list.append(loss.item())
    model.to(torch.device('cpu'))
    return np.mean(loss_list).item() #takes the mean of the evaluation loss over the batch size

class LearningProblem():
    def __init__(self, model, train_loader, optimizer, criterion, test_loader=None, test_criterion=None):
        super().__init__()
        self.model = model
        self.state_list = [copy.deepcopy(self.model.state_dict())]

        # attributes for training
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.loss_list = []
        
        # attributes for evaluation
        if test_loader is not None and test_criterion is not None:
            self.test_loader = test_loader
            self.test_criterion = test_criterion
            test_loss = evaluation_loop(self.model, self.test_loader, self.test_criterion)
            self.test_loss_list = [test_loss]
            print(f'0 epochs elapsed, evaluation loss={self.test_loss_list[-1]:.3f}')

    def train(self, epochs, progress=True, saving_step=1, subprogress=False):
        if subprogress:
            progress = False
        iterator = tqdm(range(epochs)) if progress else range(epochs)
        for i in iterator:
            loss = training_loop(self.model,
                                 self.train_loader,
                                 self.optimizer,
                                 self.criterion,
                                 progress=subprogress)
            
            self.loss_list.extend(loss)
            if (i+1) % saving_step == 0:
                self.state_list.append(copy.deepcopy(self.model.state_dict()))
            if progress:
                iterator.set_description(f'log10(loss) = {np.log10(self.loss_list[-1]):.2f}')
            elif epochs > 100 and (i+1) % (epochs // 100) == 0:
                print(f'{100 * (i+1) / epochs:.0f}% elapsed, log10(loss)={np.log10(self.loss_list[-1]):.2f}')

    def train_and_eval(self, epochs, saving_step=1, subprogress=False, averaging=False):
        assert hasattr(self, 'test_loader') and hasattr(self, 'test_criterion')
        for i in range(epochs):
            loss = training_loop(self.model,
                                 self.train_loader,
                                 self.optimizer,
                                 self.criterion,
                                 progress=subprogress,
                                 averaging=averaging)
            self.loss_list.extend(loss)
            test_loss = evaluation_loop(self.model, self.test_loader, self.test_criterion)
            self.test_loss_list.append(test_loss)
            print(f'{i+1} epochs elapsed, evaluation loss={self.test_loss_list[-1]:.3f}')
            if (i+1) % saving_step == 0:
                self.state_list.append(copy.deepcopy(self.model.state_dict()))

class LearningProblemTest():
    def __init__(self, model, train_loader, optimizer, criterion, test_loader=None, test_criterion=None):
        super().__init__()
        self.model = model
        self.state_list = [copy.deepcopy(self.model.state_dict())]
        self.saving_idx = [0]

        # attributes for training
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.loss_list = []
        self.grad_steps = 0 # number of gradient steps taken
        self.saving_step = len(self.train_loader) # number of steps between saving the model state
        
        # attributes for evaluation
        if test_loader is not None and test_criterion is not None:
            self.test_loader = test_loader
            self.test_criterion = test_criterion
            self.test_loss_list = []

            self.model.to(device)
            self.model.eval()
            loss_list = []
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                loss_list.append(self.test_criterion(outputs, targets).item())
            self.test_loss_list.append(np.mean(loss_list).item())
            self.test_idx = [0]
            print(f'evaluation loss={self.test_loss_list[-1]:.3f}')
            self.model.to(torch.device('cpu'))

    def train(self, epochs, progress=True, subprogress=False, test=False):
        if subprogress:
            progress = False
        if test:
            assert hasattr(self, 'test_loader') and hasattr(self, 'test_criterion')

        self.model.to(device)
        iterator = tqdm(range(epochs)) if progress else range(epochs)
        for epoch in iterator:

            self.model.train()
            subiterator = tqdm(self.train_loader) if subprogress else self.train_loader
            for inputs, targets in subiterator:
                inputs, targets = inputs.to(device), targets.to(device)
                self.optimizer.zero_grad()
                loss = self.criterion(inputs, targets, self.model)
                loss.backward()
                self.optimizer.step()
                self.loss_list.append(loss.item())

                if hasattr(self.model, 'clipper'):
                    self.model.clipper(self.model)

                if subprogress:
                    subiterator.set_description(f'log10(loss) = {np.log10(self.loss_list[-1]):.2f}')

                self.grad_steps += 1
                if self.grad_steps % self.saving_step == 0:
                    self.state_list.append(copy.deepcopy({k: v.cpu() for k, v in self.model.state_dict().items()}))
                    self.saving_idx.append(self.grad_steps)

            if test:
                self.model.eval()
                loss_list = []
                for inputs, targets in self.test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = self.model(inputs)
                    loss_list.append(self.test_criterion(outputs, targets).item())
                self.test_loss_list.append(np.mean(loss_list).item())
                self.test_idx.append(self.grad_steps)

            if progress:
                iterator.set_description(f'log10(loss) = {np.log10(self.loss_list[-1]):.2f}')
            elif epochs > 100 and (epoch+1) % (epochs // 100) == 0:
                print(f'{100 * (epoch+1) / epochs:.0f}% elapsed, log10(loss)={np.log10(self.loss_list[-1]):.2f}')
        
        self.model.to(torch.device('cpu'))
                

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
    def __init__(self, input_dim, width, activation, projection=nn.Identity()):
        super().__init__()
        self.weight = nn.Parameter(data=torch.randn(width, input_dim), requires_grad=True)
        self.activation = activation
        self.projection = projection

    def forward(self, x):
        return self.activation(torch.linalg.vector_norm(self.projection(self.weight.T[None,:,:] - x[:,:,None]), dim=1))

## Models constructors
def SHL(input_dim, width, activation, bias=False, VarProTraining=True, clipper=None):
    feature_model = SHLFeatureModel(input_dim, width, activation, bias=bias)
    return VarProModel(feature_model, width, 1, VarProTraining=VarProTraining, clipper=clipper)

def Convolution(input_dim, width, activation, projection=nn.Identity(), VarProTraining=True, clipper=None):
    feature_model = ConvolutionFeatureModel(input_dim, width, activation, projection=projection)
    return VarProModel(feature_model, width, 1, VarProTraining=VarProTraining, clipper=clipper)


## Criterions
# perform projection on the outer layer
class ExactRidgeProjection():
    def __init__(self, lmbda, momentum=None):
        self.lmbda = lmbda
        if momentum is not None and not (momentum >= 0 and momentum < 1):
            raise ValueError('momentum should be between 0 and 1')
        self.momentum = momentum

    @torch.no_grad()
    def __call__(self, inputs, targets, model, requires_grad=False):
        features = model.feature_model(inputs).clone().detach()
        batch_size, width = features.shape[0], features.shape[1]
        if batch_size > width: ## underparameterized case
            K = (features.T @ features) / (batch_size * width)
            u = torch.linalg.solve(K + self.lmbda * torch.eye(width).to(K.device),  targets.T @ features / batch_size, left=False)
        else: ## overparameterized case
            K = (features @ features.T) / (batch_size*width)
            u = torch.linalg.solve(K + self.lmbda * torch.eye(batch_size).to(K.device), targets.T, left=False) @ features / batch_size
        if self.momentum is not None:
            u = self.momentum * model.outer.weight + (1-self.momentum) * u
        model.outer.weight = nn.Parameter(data=u, requires_grad=requires_grad)

# perform projection on the outer layer with the unbiasing -1 term
class ExactRidgeProjectionUnbiased():
    def __init__(self, lmbda):
        self.lmbda = lmbda
    
    @torch.no_grad()
    def __call__(self, inputs, targets, model, requires_grad=False):
        features = model.feature_model(inputs).clone().detach()
        batch_size, width = features.shape[0], features.shape[1]
        K = (features.T @ features) / (batch_size * width)
        u = torch.linalg.solve(K + self.lmbda * torch.eye(width).to(K.device),  (targets.T @ features) / batch_size + self.lmbda, left=False)
        model.outer.weight = nn.Parameter(data=u, requires_grad=requires_grad)

# least square criterion with projection
class VarProCriterion(nn.Module):
    def __init__(self, lmbda, num_classes=None, momentum=None):
        super().__init__()
        self.lmbda = lmbda
        self.projection = ExactRidgeProjection(lmbda=lmbda, momentum=momentum)
        self.num_classes = num_classes
        
    def forward(self, inputs, targets, model):
        if self.num_classes is not None:
            targets = nn.functional.one_hot(targets, num_classes=self.num_classes).to(device=inputs.device, dtype=inputs.dtype)
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
        return 0.5 * ((predictions - targets)**2).mean() / self.lmbda + 0.5 * ((model.outer.weight-1)**2).mean()
    

# Classification loss with projection
class ClassifAccuracy(nn.Module):
    def __init__(self, num_classes, top_k=None):
        super().__init__()
        self.num_classes = num_classes
        if top_k is None:
            print('top_k is None: Using top-1 accuracy as default')
            self.criterion = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, top_k=1).to(device)
        else:
            self.criterion = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(device)
        
    def forward(self, inputs, targets, model):
        predictions = model(inputs) # MulticlasAccuracy will automatically apply argmax
        return self.criterion(predictions, targets)
    
# least square criterion with general regularization
class TwoTimescaleCriterion(nn.Module):
    def __init__(self, lmbda, regularization_function, num_classes=None):
        super().__init__()
        self.lmbda = lmbda
        self.regularization_function = regularization_function
        self.num_classes = num_classes
        
    def forward(self, inputs, targets, model):
        if self.num_classes is not None:
            targets = nn.functional.one_hot(targets, num_classes=self.num_classes).to(device=inputs.device, dtype=inputs.dtype)
        predictions = model(inputs)
        return 0.5 * ((predictions - targets)**2).mean() / self.lmbda + self.regularization_function(model.outer.weight).mean()
    
## Regularization functions
class power_regularization():
    def __init__(self, p=2):
        self.p = p
        
    def __call__(self, x):
        return torch.abs(x)**self.p / self.p
    
class power_regularization_unbiased():
    def __init__(self, p=2):
        self.p = p
        
    def __call__(self, x):
        return torch.abs(x-1)**self.p / self.p

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
