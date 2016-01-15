#from __future__ import division
import numpy as np
from theano import function, config
import theano.tensor as T
import cPickle as pk

'''
helper functions
'''

floatX = config.floatX

# XXX dataset parameters
MNIST_PATH = '../mnist.pkl.gz'  ##data/mnist.pkl.gz
FREY_PATH = '../freyfaces.pkl'

def load_dataset(dset):  ##='mnist'
    if dset == 'mnist':
        import gzip
        f = gzip.open(MNIST_PATH, 'rb')
        train_set, valid_set, test_set = pk.load(f)
        f.close()
        data = {'train': train_set, 'valid': valid_set, 'test': test_set}
    elif dset == 'freyfaces':
        with open(FREY_PATH,'rb') as f:
            data = pk.load(f)
            data = np.asarray(data, dtype=floatX)
            f.close()
    else:
        raise RuntimeError('unrecognized dataset: %s' % dset)
    return data

# costs
def kld_unit_mvn(mu, var):  # KL divergence from N(0, I) ##iclr14AEVBappend.B, -KL
    return (mu.shape[1] + T.sum(T.log(var), axis=1) - T.sum(T.square(mu), axis=1) - T.sum(var, axis=1)) / 2.0

# costs, covC=D+u*u.T, icml14SBP (13) (19) (20)
def kldu_unit_mvn(mu, var, u):
    eta = 1.0/(1.0+T.sum(u*u/var, axis=1))  ##icml14SBP(20):log|C|=log(eta)-log(var)
    return (mu.shape[1] + T.sum(T.log(var), axis=1) - T.log(eta) - T.sum(T.square(mu), axis=1) - T.sum(var+u*u, axis=1)) / 2.0

def log_diag_mvn(mu, var):
    def f(x):  #expects data-batches, 2nd part of iclr14AEVB(10)
        k = mu.shape[1]
        logp = (-k/2.0)*np.log(2*np.pi) - 0.5*T.sum(T.log(var), axis=1) - T.sum(0.5*(1.0/var)*(x-mu)*(x-mu), axis=1)
        return logp
    return f  ##return a function handle!

def log_nondiag_mvn(mu, var, u):  ##icml14SBP (13) (20)
    def f(x):
        k = mu.shape[1]
        eta = 1.0/(1.0+T.sum(u*u/var, axis=1))
        logp = (-k/2.0)*np.log(2*np.pi) - 0.5*T.sum(T.log(var), axis=1) + 0.5*T.log(eta) - T.sum(0.5*(x-mu)*(x-mu)/var, axis=1) + eta*T.sqr( T.sum((x-mu)*u/var, axis=1) )
        return logp
    return f

# test things out
if __name__ == '__main__':
    f = log_diag_mvn(np.zeros(2), np.ones(2))
    x = T.vector('x')
    g = function([x], f(x))
    print g(np.zeros(2))
    print g(np.random.randn(2))

    mu = T.vector('mu')
    var = T.vector('var')
    j = kld_unit_mvn(mu, var)
    g = function([mu, var], j)
    print g(np.random.randn(2), np.abs(np.random.randn(2)))
