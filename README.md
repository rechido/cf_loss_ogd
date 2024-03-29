## Orthogonal Gradient Descent based Catastrophic Forgetting Loss

An Implementation of CF loss-based OGD for continual learning in PyTorch.

Paper Title: A Continual Learning Algorithm Based on Orthogonal Gradient Descent Beyond Neural Tangent Kernel Regime

Journal: IEEE Access (Published at 09 Aug 2023)

Author: Da Eun Lee, Kensuke Nakamura, Jae-Ho Tak, and Byung-Woo Hong

Affiliation: Department of Artificial Intelligence, Chung-Ang University, Seoul, Korea

URL: https://ieeexplore.ieee.org/abstract/document/10213447

## Experiments

The experiments have done on Split-MNIST, Permuted-MNIST, Split-CIFAR100.

The experimental results are presented below.

The numbers in the results represent the average accuracy, which is a common metric widely used in continual learning as follows:
<p align="center">
  <img src="resources/equation23.png" />
</p>
where a_ij is the accuracy of the j-th test set after completing the learning of the i-th training set.


1. Characteristic Investigation (Our method)
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

Run 'script.sh' file as follows:
```
bash script.sh
```
Please uncomment the line you want to run.

You may need to download the benchmark datasets at the designated path before running.

Check various options given in the script file as examples for other experimental setups.

## References

Our code is based on the preceding studies as follows:

Code
https://github.com/tldoan/PCA-OGD

Paper
1. http://proceedings.mlr.press/v108/farajtabar20a/farajtabar20a.pdf
2. http://proceedings.mlr.press/v130/doan21a/doan21a.pdf
