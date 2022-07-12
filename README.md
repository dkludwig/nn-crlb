# nn-crlb

## Motivation
This repository has code that (at a high level) models the data acquisition and reconstruction processes in magnetic resonance imaging (MRI). The basic problem is as follows: in order to speed up MRI scanning times, parallel imaging methods use multiple scanning coils simultaneously to compensate for undersampling the k-space data. Each of these scanning coils outputs a different "channel" of measurements, which can be understood as linear functions of the underlying position-space structure. However, the sensitivity maps that contribute to these functions have a complex dependence on the geometry of the measurement apparatus and are therefore not known independently. Deep learning methods are becoming more common in the effort to accurately invert this unknown linear encoding (e.g. the [fastMRI](https://fastmri.org/) competition), but their inherent opacity leads to concerns in their noise propagation properties.


## Model
To model this scenario, the `make_data.py` script generates a random, full-rank 100x500 matrix (the data acquisition function) and 50,000 random 100-element vectors (the position-space data), and then encodes the vectors with the matrix to get the 500-element training vectors (the k-space data). The `train_network.py` script trains a 50,000x5,000x100 fully-connected neural network using TensorFlow to reconstruct the original vectors from the training vectors.

The trained network is then analyzed in the notebook `nn-analysis.ipynb` by comparing the analytical Cramer-Rao lower bound on the variance of statistical estimators to Monte Carlo noise simulations.

### CRLB
(Add citations to both stat books)

Have random vector $\vec X \sim f(\vec x;\vec \theta)$ in $\mathbb{R}^n$, estimator $\hat{\vec\theta}:\mathbb{R}^n \rightarrow \mathbb{R}^d$ s.t. $E(\hat{\vec\theta})=\vec g (\vec\theta)$. For this part of the derivation, let all vectors be column vectors and gradients be row vectors (unfortunately this will have to change later on in order to match Tensorflow's syntax, but for now it's easier). Differentiating the definition of $\vec g$:

$$\vec g (\vec\theta) = \int_{\mathbb{R}^n} \hat{\vec\theta}(\vec x) f(\vec x; \vec\theta) d^n\vec x
\Rightarrow \frac{\partial \vec g}{\partial \vec\theta} = \int_{\mathbb{R}^n} \hat{\vec\theta}(\vec x) \frac{\partial f}{\partial \vec\theta}(\vec x; \vec\theta)d^n\vec x$$

Inside the second integral is a column vector multiplying a row vector a.k.a. a dxd tensor product. Since $\partial \log{f} / \partial \theta_i = (\partial f/ \partial \theta_i)/f$, have

$$ \frac{\partial \vec g}{\partial \vec\theta} = \int_{\mathbb{R}^n} [\hat{\vec\theta}(\vec x) \frac{\partial \log{f}}{\partial \vec\theta}(\vec x; \vec\theta)]f(\vec x; \vec\theta) d^n\vec x
= E(\hat{\vec\theta} \vec Z)$$


where we've defined a new random variable $\vec Z = (\partial \log{f}/\partial \vec\theta)(\vec X;\vec\theta)$. The normalization of $f$ implies that the expected value of $\vec Z$ is $\vec 0$:

$$ \int_{\mathbb{R}^n} f(\vec x; \vec\theta)d^n\vec x = 1 \Rightarrow
E(\vec Z) = \int_{\mathbb{R}^n} \frac{\partial \log{f}}{\partial \vec\theta}(\vec x; \vec\theta)f(\vec x; \vec\theta) d^n\vec x =
\frac{\partial}{\partial \vec\theta}  \int_{\mathbb{R}^n} f(\vec x; \vec\theta) d^n\vec x = \vec 0$$

This implies that

$$ \text{Cov}(\hat{\vec\theta},\vec Z) = E(\hat{\vec\theta} \vec Z) = \frac{\partial \vec g}{\partial \vec\theta}$$

Cauchy-Schwartz ensures that the correlation coefficient satisfies $-1 \leq \rho \leq 1$, so

$$[\frac{\partial \vec g}{\partial \vec\theta}]_{ii}^2 = \text{Cov}(\hat{\theta_i},Z_i)
= \rho_{ii}^2 \text{Var}(\hat{\theta_i}) \text{Var}(Z_i)
\leq \text{Var}(\hat{\theta_i}) [\mathbf{I}(\vec\theta)]_{ii}$$

where $[\mathbf{I}(\vec\theta)]_{ij}=E(Z_i Z_j) = E(-\partial Z_i / \partial \theta_j)$ is the Fisher information matrix (the second equality follows from differentiation the normalization condition again). Rearranging,

$$ \text{Var}(\hat{\theta_i}) \geq \frac{[\frac{\partial \vec g}{\partial \vec\theta}]_{ii}^2}{[\mathbf{I}(\vec\theta)]_{ii}}$$
