# ResNets for classification on CIFAR10 (Section 6.2)

In the paper we consider:

* Training a ResNet with a stochastic variant of VarPro:
```
python CIFAR10_VarProM.py --lmbda 1e-3 --batch_size 64 --time_scale 1e-3 --momentum 0.9 --epochs 100
```
for `--batch_size` in `{64, 128, 256, 512, 1024}`.
* Training a ResNet with SGD with momentum:
```
python CIFAR10_SGDM.py --lmbda 1e-3 --batch_size 64 --time_scale 1e-3 --momentum 0.9 --epochs 100
```
for `--batch_size` in `{64, 128, 256, 512, 1024}`.
* Training a ResNet with Shampoo:
```
python CIFAR10_Shampoo.py --lmbda 1e-3 --batch_size 64 --time_scale 1e-2 --momentum 0.9 --epochs 100
```
for `--batch_size` in `{64, 128, 256, 512, 1024}`.

Results will be stored in `results`.

Once experiments are perfomed, the figures in Section 6.2 can be reproduced with the notebook `Results_resnets.ipynb`.