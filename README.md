# Verification_Guided_Control_Learning

This repository contains the main implementation of "Verification in the Loop: Correct-by-Construction Control Learning with Reach-avoid Guarantees", which is submitted to NeurIPS 2021.  

## Installation
Please refer to [ReachNN](https://github.com/JmfanBU/ReachNNStar) to set up the environment. 

## Usage

To run the controller synthesis for Adaptive Crucise Control(ACC), please do
```shell
./example_acc.sh
```

To run the controller synthesis for non-linear Van der Pol's Oscillator system, please do
```
cd Bernstein_Polynomial_Approximation
make name=nn_os_relu_tanh
python controller_approximation_lib.py -flag 1
```

Please consider to use run.py to automically run multiple times for the experiments.

## Contributing 
Please open an issue if there is any problem you encounter with.

## To do
Add more systems into the experiments.