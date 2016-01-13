# for mnist or freyfaces data viewing
from theano import config
import gzip
import numpy as np
import cPickle as pk
import matplotlib.pyplot as plt

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-dset', default='mnist') ##mnist, freyfaces
    parser.add_argument('-start', default=0, type=int)
    parser.add_argument('-width', default=50, type=int)
    parser.add_argument('-height', default=30, type=int)
    args = parser.parse_args()

    if args.dset == 'mnist':
        with gzip.open('mnist.pkl.gz', 'rb') as f:
            (x_train, t_train), (x_valid, t_valid), (x_test, t_test) = pk.load(f)
            f.close()
        S = (28, 28)
        data = x_train
    elif args.dset == 'freyfaces':
        with open('freyfaces.pkl', 'rb') as f:
            data = pk.load(f)
            f.close()
        S = (28,20)
    
    print "shape: ", data.shape
    start=args.start
    h=args.height
    w=args.width

    dview = np.zeros((S[0]*h, S[1]*w), dtype=config.floatX)
    for z1 in xrange(h):
        for z2 in xrange(w):
            x_hat = data[start+z2+w*z1].reshape(S)
            dview[z1*S[0]:(z1+1)*S[0], z2*S[1]:(z2+1)*S[1]] = 1-x_hat

    plt.imshow(dview, cmap='Greys_r')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
