# Reservoir Computing Overview

Reservoir Computing (RC) is a framework for training Recurrent Neural Networks (RNNs) where the recurrent part (the "reservoir") is fixed, and only the output weights are trained. This allows for extremely fast training compared to traditional Backpropagation Through Time (BPTT).

## Echo State Networks (ESN)

The standard Echo State Network (ESN) update equation is:

$$ \mathbf{x}(t+1) = (1-\alpha)\mathbf{x}(t) + \alpha \tanh(\mathbf{W}_{in}\mathbf{u}(t+1) + \mathbf{W}_{res}\mathbf{x}(t)) $$

Where:
*   $\mathbf{x}(t)$ is the reservoir state vector.
*   $\mathbf{u}(t)$ is the input vector.
*   $\mathbf{W}_{in}$ is the input weight matrix.
*   $\mathbf{W}_{res}$ is the reservoir weight matrix.
*   $\alpha$ is the leaking rate ($\alpha \in (0, 1]$).
