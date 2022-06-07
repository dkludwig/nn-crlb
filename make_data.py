'''
Generate and encode data for neural network to invert.

The training data are 50,000 random 100-element vector
(with elements sampled from independent uniform distributions)
that are encoded by a random, full-rank, 100x500 matrix A
(with elements sampled from independent standard normal distributions)
into 500-element vectors. The test data are 1,000 more vectors 
generated and encoded in the same way. 

The training data, test data, and encoding matrix A are all saved to
"nn-inverse-data.npz"
'''

import numpy as np

def main():
    # samples from uniform(0,1), each data point is a row vector
    # (TensorFlow likes having the first dimension of the array
    # index separate data points / observations)
    y_train = np.random.rand(50000, 100)
    y_test = np.random.rand(1000, 100)

    # generate full-rank encoding matrix A
    is_full_rank = False
    while(not is_full_rank):
        # normal random variables
        A = np.random.randn(100, 500)
        is_full_rank = (np.linalg.matrix_rank(A) == 100)

    # encode y_train and y_test by right-multiplying A
    # x.shape = (N, 100), A.shape = (100, 500) -> output shape of (N, 500)
    x_train = y_train @ A
    x_test = y_test @ A

    # save original and encoded data to .npz archive
    np.savez('nn-inverse-data.npz',
             x_train=x_train,
             x_test=x_test,
             y_train=y_train,
             y_test=y_test,
             A=A)
    
if __name__ == '__main__':
    main()
