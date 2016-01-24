THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python vae.py -epochs 4000 -hdim 500 -COV True -outfile mnist_model.pk -dset mnist
python viewM.py mnist_model.pk mnist

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python vae.py -epochs 1000 -hdim 200 -COV True -outfile freyfacesM.pk -dset freyfaces
python viewM.py freyfacesM.pk freyfaces
