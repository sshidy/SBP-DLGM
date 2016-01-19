#from __future__ import division
import numpy as np
from theano import config, shared, scan, function
import theano.tensor as T
from utils import log_diag_mvn, log_nondiag_mvn, floatX

rng = np.random.RandomState(1234)
SMALLNUM = 1e-20
LEAKY = 1e-6

class HiddenLayer(object):
    def __init__(self, input, n_in, n_out, W=None, b=None, activation=T.tanh, prefix='', stoch=False, G=None):
        self.n_in = n_in
        self.n_out = n_out

        if W is None:
            # glorot init vs randn, glorot worked better after 1 epoch with adagrad
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                #rng.randn(n_in, n_out) * 0.01,
                dtype=floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = shared(value=W_values, name=prefix+'_W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=floatX)
            b = shared(value=b_values, name=prefix+'_b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        # self.output = (
            # lin_output if activation is None
            # else activation(lin_output)  ##activation(lin_output)
        # )
        if activation == None:
            self.output = lin_output
        elif activation == T.nnet.relu:
            self.output = T.nnet.relu(lin_output, LEAKY)
        else:
            self.output = activation(lin_output)
        
        self.params = [self.W, self.b]
        
        if stoch==True:
            if G==None:
                G_values = np.asarray(rng.randn(n_out, n_out)*0.01, dtype=floatX)
                G = shared(value=G_values, name=prefix+'_G', borrow=True)
                self.G = G
                self.params = self.params + [self.G]
            ksi = np.asarray(rng.randn(n_out), dtype=floatX)
            self.output = self.output + T.dot(self.G, ksi)
        

class _MLP(object):  # building block for MLP instantiations
    def __init__(self, x, n_in, n_hid, nlayers=1, prefix=''):  ##G=None, Lksi=None, 
        self.nlayers = nlayers
        self.hidden_layers = list()
        inp = x
        # if Lksi:
            # if G is None:
                # ENCODER = True
                # G_values = np.asarray(rng.randn(nlayers, n_hid, n_hid)*0.01, dtype=floatX)
                # G = shared(value=G_values, name=prefix+'_G', borrow=True)
                # self.G = G
                # self.params = self.params + [self.G]

        ##nlayers before output
        for k in xrange(self.nlayers):  ##input->h, require tanh()
            hlayer = HiddenLayer(
                input=inp,
                n_in=n_in,
                n_out=n_hid,
                activation=T.tanh,
                prefix=prefix + ('_%d' % (k + 1))
            )
            n_in = n_hid
            inp = hlayer.output
            # if ENCODER == True:
                # inp = inp + T.dot(self.G[k], Lksi[k])
            # else:  #decoder
                # inp = inp + Lksi[nlayers-k+1]
                
            self.hidden_layers.append(hlayer)
            self.params = [param for l in self.hidden_layers for param in l.params]
        self.input = input

class GaussianMLP(_MLP):
    def __init__(self, x, n_in, n_hid, n_out, nlayers=1, activation=None, y=None, eps=None, COV=False):  ##Lksi=None, 
        # if Lksi:
            # if eps and (y is None): #encoder!
                # super(GaussianMLP, self).__init__(x, n_in, n_hid, nlayers=nlayers, prefix='GaussianMLP_hidden', Lksi)
            # elif (eps is None) and y: #decoder!
                # super(GaussianMLP, self).__init__(x, n_in, n_hid, nlayers=nlayers, prefix='GaussianMLP_hidden', Lksi, self.G)
        # else:
        super(GaussianMLP, self).__init__(x, n_in, n_hid, nlayers=nlayers, prefix='GaussianMLP_hidden')
                
        ##mu&logvar are affine from h when encode
        self.mu_layer = HiddenLayer(
            input=self.hidden_layers[-1].output,
            n_in=self.hidden_layers[-1].n_out,
            n_out=n_out,
            activation=activation,  ##None T.nnet.softplus, not much diff. if logvar>0 freyfaces
            prefix='GaussianMLP_mu'
        )
        # log(sigma^2)  ##h generate logvar, not sigma! logvar=2logsigma
        self.logvar_layer = HiddenLayer(
            input=self.hidden_layers[-1].output,
            n_in=self.hidden_layers[-1].n_out,
            n_out=n_out,
            activation=activation,  ##None, ReLU|sigmoid, keep logvar>0 for freyfaces
            prefix='GaussianMLP_logvar'
        )
        self.mu = self.mu_layer.output
        self.var = T.exp(self.logvar_layer.output)
        self.params = self.params + self.mu_layer.params + self.logvar_layer.params
        
        def SampleKsi(d, u, mu, eps):  # icml14SBP(20)
            dn = 1.0/d
            uDnu = T.sum(u*u*dn)
            coeff = ( 1-1.0/T.sqrt(1.0+uDnu) ) / (uDnu+SMALLNUM)
            u = u.reshape((u.shape[0],1))
            R = T.diag(T.sqrt(dn)) - coeff*T.dot( T.dot(T.diag(dn),T.dot(u,u.T)), T.diag(T.sqrt(dn)) )
            return mu + T.dot(R,eps)

        if COV == False:
            self.sigma = T.sqrt(self.var)
            if eps:  # for use as encoder
                assert(y is None)
                self.out = self.mu + self.sigma * eps
            if y:  # for use as decoder
                assert(eps is None)
                self.out = T.nnet.sigmoid(self.mu)  ##the grey degree of each pixel
                #Gaussian-LL of data y under (z, params)
                self.cost = -T.sum(log_diag_mvn(self.out, self.var)(y))  ##(self.out, self.var)
        else:
            self.cov_u_layer = HiddenLayer(
                input=self.hidden_layers[-1].output,
                n_in=self.hidden_layers[-1].n_out,
                n_out=n_out,
                activation=activation,
                prefix='GaussianMLP_cov_u'
            )
            self.u = self.cov_u_layer.output
            self.params = self.params + self.cov_u_layer.params
            if eps:  ##icml14(21)
                assert(y is None)
                self.out, _ = scan(SampleKsi, sequences=[self.var, self.u, self.mu, eps])
            if y:  # for use as decoder
                assert(eps is None)
                self.out = T.nnet.sigmoid(self.mu)  ##the grey degree of each pixel
                self.cost = -T.sum(log_nondiag_mvn(self.mu, self.var, self.u)(y))
            
class BernoulliMLP(_MLP):

    def __init__(self, x, n_in, n_hid, n_out, nlayers=1, y=None):
        super(BernoulliMLP, self).__init__(x, n_in, n_hid, nlayers=nlayers, prefix='BernoulliMLP_hidden')
        self.out_layer = HiddenLayer(
            input=self.hidden_layers[-1].output,
            n_in=self.hidden_layers[-1].n_out,
            n_out=n_out,
            activation=T.nnet.sigmoid,  ##T.nnet.sigmoid
            prefix='BernoulliMLP_y_hat'
        )
        self.params = self.params + self.out_layer.params
        if y:
            self.out = self.out_layer.output
            self.cost = T.sum(T.nnet.binary_crossentropy(self.out, y))
