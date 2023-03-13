# Physics-informed Neural Operator for 2D wave equation
More information about the project is in [pino-report.pdf](pino-report.pdf).
## Installation
### Use a conda environment with Modulus-22.09
```bash
conda create -f modulus-env.yaml
conda activate modulus-22.09
```
### Optional to view the results
```bash
conda install tensorboard
```
## Configurations
Set the parameters in [config.yaml](config.yaml).

## Train the model
```bash
python3 wave2d.py
```
## View Metrics
```bash
tensorboard --logdir=outputs/
```

