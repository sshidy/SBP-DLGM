import numpy as np
from theano import function, shared
import theano.tensor as T
import cPickle as pk
from mlp import GaussianMLP, BernoulliMLP
from utils import kld_unit_mvn, kldu_unit_mvn, load_dataset, floatX
import time

ADAG_EPS = 1e-12  # for stability ##1e-10

class VAE(object):

    def __init__(self, xdim, args, dec_nonlin=None):
        self.xdim = xdim
        self.hdim = args.hdim
        self.zdim = args.zdim
        self.lmbda = args.lmbda  # weight decay coefficient * 2
        self.x = T.matrix('x', dtype=floatX)
        self.eps = T.matrix('eps', dtype=floatX)
        self.train_i = T.scalar('train_i', dtype=floatX)
        self.dec = args.decM
        self.COV = args.COV

        self.enc_mlp = GaussianMLP(self.x, self.xdim, self.hdim, self.zdim, nlayers=args.nlayers, eps=self.eps, COV=self.COV)
        if self.dec == 'bernoulli':
            # log p(x | z) defined as -CE(x, y) = dec_mlp.cost(y)
            self.dec_mlp = BernoulliMLP(self.enc_mlp.out, self.zdim, self.hdim, self.xdim, nlayers=args.nlayers, y=self.x)
        elif self.dec == 'gaussian':
            self.dec_mlp = GaussianMLP(self.enc_mlp.out, self.zdim, self.hdim, self.xdim, nlayers=args.nlayers, y=self.x, activation=dec_nonlin, COV=self.COV)
        else:
            raise RuntimeError('unrecognized decoder %' % dec)
        #encoder part + decoder part
        if self.COV == False:
            self.enc_cost = -T.sum(kld_unit_mvn(self.enc_mlp.mu, self.enc_mlp.var))
        else:
            self.enc_cost = -T.sum(kldu_unit_mvn(self.enc_mlp.mu, self.enc_mlp.var, self.enc_mlp.u))
        self.cost = (self.enc_cost + self.dec_mlp.cost) / args.batsize
        self.params = self.enc_mlp.params + self.dec_mlp.params
        ##[T.grad(self.cost, p) + self.lmbda * p for p in self.params]
        self.gparams = [T.grad(self.cost, p) for p in self.params]
        self.gaccums = [shared(value=np.zeros(p.get_value().shape, dtype=floatX)) for p in self.params]
        self.lr = args.lr * (1-args.lmbda)**self.train_i

        # update params, update sum(grad_params) for adagrade
        self.updates = [
                (param, param - self.lr*gparam/T.sqrt(gaccum+T.square(gparam)+ADAG_EPS))
                for param, gparam, gaccum in zip(self.params, self.gparams, self.gaccums) ]
        self.updates += [ (gaccum, gaccum + T.square(gparam))
                    for gaccum, gparam in zip(self.gaccums, self.gparams)  ]

        self.train = function(
            inputs=[self.x, self.eps, self.train_i],
            outputs=self.cost,
            updates=self.updates
        )
        self.test = function(
            inputs=[self.x, self.eps],
            outputs=self.cost,
            updates=None
        )
        # can be used for semi-supervised learning for example
        self.encode = function(
            inputs=[self.x, self.eps],
            outputs=self.enc_mlp.out
        )
        # use this to sample
        self.decode = function(
            inputs=[self.enc_mlp.out],  ##z with shape (1,2)
            outputs=self.dec_mlp.out
        ) ##mlp103 .out=.mu+.sigma*eps
        

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-batsize', default=100)  ##100
    parser.add_argument('-nlayers', default=1, type=int, help='num_hid_layers before output')
    parser.add_argument('-hdim', default=200, type=int) ##200 for freyfaces
    parser.add_argument('-zdim', default=2, type=int)  ##2
    parser.add_argument('-lmbda', default=0., type=float, help='weight decay coeff') ##0.001
    parser.add_argument('-lr', default=0.01, type=float, help='learning rate')  ##0.01
    parser.add_argument('-epochs', default=100, type=int)  ##1000
    parser.add_argument('-print_every', default=5, type=int)  ##100
    parser.add_argument('-save_every', default=50, type=int)  ##1
    parser.add_argument('-outfile', default='vae_model.pk')
    parser.add_argument('-dset', default='mnist') ##mnist freyfaces
    parser.add_argument('-COV', default=False, type=bool)
    parser.add_argument('-decM', default='gaussian', help='bernoulli | gaussian')
    args = parser.parse_args()

    batsize = args.batsize
    dset = args.dset
    data = load_dataset(dset)
    valid_fg = 0
    dec_nonlin = T.nnet.relu  ##T.nnet.softplus
    if dset=='mnist':
        train_x, train_y = data['train']  ##mnist: (N,784)
        valid_x, valid_y = data['valid']
        num_valid_bats = valid_x.shape[0] / batsize
        print "valid data shape: ", valid_x.shape
        valid_fg = 1
    elif dset=='freyfaces':
        train_x = data
    print "training data shape: ", train_x.shape

    model = VAE(train_x.shape[1], args, dec_nonlin=dec_nonlin)

    num_train_bats = train_x.shape[0] / batsize  ##discard last <batsize

    begin = time.time()
    for i in xrange(args.epochs):
        for k in xrange(num_train_bats):
            x = train_x[k*batsize : (k+1)*batsize, :]
            eps = np.random.randn(x.shape[0], args.zdim).astype(floatX)
            cost = model.train(x, eps, i)  ##update_times=epochs*num_train_bats
        j = i+1
        if j % args.print_every == 0:  ##(b+1)
            end = time.time()
            print('epoch %d, cost %.2f, time %.2fs' % (j, cost, end-begin))
            begin = end
            if valid_fg == 1:
                valid_cost = 0
                for l in xrange(num_valid_bats):
                    x_val = valid_x[l*batsize:(l+1)*batsize, :]
                    eps_val = np.zeros((x_val.shape[0], args.zdim), dtype=floatX)
                    valid_cost = valid_cost + model.test(x_val, eps_val)
                valid_cost = valid_cost / num_valid_bats
                print('valid cost: %f' % valid_cost)
        if j % args.save_every == 0:  ##
            with open(args.outfile, 'wb') as f:
                pk.dump(model, f, protocol=pk.HIGHEST_PROTOCOL)
            print('model saved')

    # with open(args.outfile, 'wb') as f:
        # pk.dump(model, f, protocol=pk.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
