# nn-crlb
This repository has code that (at a high level) models the data acquisition and reconstruction processes in magnetic resonance imaging (MRI). The basic problem is as follows: in order to speed up MRI scanning times, parallel imaging methods use multiple scanning coils simultaneously to compensate for under-sampling of the k-space data. While the multiple channels of k-space data can be understood as linear functions of the underlying position-space structure, the sensitivity maps of each coil that contribute to these functions have a complex dependence on the geometry of the measurement apparatus and are therefore not known independently. Deep learning methods are becoming more common in the effort to accurately invert this unknown linear encoding (e.g. the [fastMRI](https://fastmri.org/) competition), but their inherent opacity understandably leads to concerns in their noise propagation properties.

To model this scenario, the `make_data.py` script generates a random, full-rank 100x500 matrix (the data acquisition function) and 50,000 random 100-element vectors (the position-space data), and then encodes the vectors with the matrix to get the 500-element training vectors (the k-space data). The `train_*_network.py` script(s) train a 50,000x5,000x100 fully-connected neural network using TensorFlow to reconstruct the original vectors from the training vectors.

The trained network is then analyzed in the notebook `nn-analysis.ipynb` by comparing the Cramer-Rao lower bound on the variance of statistical estimators to Monte Carlo noise simulations. [More exposition coming]
