# Convolutional-Neural-Network
* Base on: https://github.com/SergiosKar/Convolutional-Neural-Network

Development of computer vision framework based on deep learning and  GPU programming. We designed a convolutional and a fully connected neural network for object recognition in images. The training of the network and the computation of its output run on GPU, programmed with OpenCL and C++. The algorithm developed during a master thesis.

For more details check my blog posts:

https://sergioskar.github.io/Neural_Network_from_scratch/

https://sergioskar.github.io/Neural_Network_from_scratch_part2/

# Issue
* [ld: warning: dylib was built for newer macOS version (11.3) than being linked (11.1)](https://stackoverflow.com/questions/71112682/ld-warning-dylib-was-built-for-newer-macos-version-11-3-than-being-linked-1)
    * `export MACOSX_DEPLOYMENT_TARGET=11.3` can eliminate the warning.


# Preparation
* [Download Cifar-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
    * CIFAR-10 binary version (suitable for C programs)
    * `tar -xzvf cifar-10-binary.tar.gz`

# Quick Start
```
make
./my_program
```