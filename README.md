## Orthogonal Gradient Descent based Catastrophic Forgetting Loss

An Implementation of CF loss-based OGD for continual learning in PyTorch.

Paper Title: A Continual Learning Algorithm Based on Orthogonal Gradient Descent Beyond Neural Tangent Kernel Regime

Journal: IEEE Access (Accepted at 30 Jul 2023, to be published soon)

## Experiments

The experiments have done on Split-MNIST, Permuted-MNIST, Split-CIFAR100.

The experimental results are presented below.

1. Characteristic Investigation
<p align="center">
  <img src="resources/figure1.png" />
</p>

2. Architectural Investigation
<p align="center">
  <img src="resources/figure3.png" />
</p>

3. Comparative experiments
<p align="center">
  <img src="resources/table5.png" />
</p>

## How to use

Run 'script.sh' file through desired settings as follows:
```
bash script.sh
```
You may need to download the benchmark datasets at the designated path before running.

Check various options in the script file for other experimental setups.
