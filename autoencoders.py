from __future__ import print_function

import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class DenoisingAutencoder(object):
    def __init__(self,
                 numpy_rng,
                 theano_rng=None,
                 input=None,
                 n_visible =784,
                 n_hidden=500,
                 W=None,
                 bhid=None,
                 bvis=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2**30))

        if not W:
            initial_W =  np.asarray(numpy_rng.uniform
                                    (low=-4*np.sqrt(6./(n_hidden+n_visible)),
                                                    high =-4*np.sqrt(6./(n_hidden+n_visible)),
                                     size = (n_visible,n_hidden)),
                                    dtype=theano.config.floatX
                                    )
            W= theano.shared(value=initial_W,name='W',borrow=True)
        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_hidden_values(self,input):
        return T.nnet.sigmoid(T.dot(input,self.W)+self.b)

    def get_reconstructed_input(self,hidden):
        return T.nnet.sigmoid(T.dot(hidden,self.W_prime)+self.b_prime)

    def get_corrupted_input(self,input,corruption_level):
        #simple random masking corruption
        return self.theano_rng.binomial(size=input.shape,
                                        n=1,
                                        p=1-corruption_level,
                                        dtype=theano.config.floatX)



    def get_cost_updates(self,corruption_level,learning_rate):

        tilde_x = self.get_corrupted_input(self.x,corruption_level)
        y= self.get_hidden_values(tilde_x)
        z=self.get_reconstructed_input(y)

        #calculate the cross-entropy loss of the input x and the reconstructed
        #ouput x
        L = -T.sum(self.x *T.log(z) + (1-self.x))
        # L is now a vector where each element is teh cost of corresponding ex
        #need to compute the average of these to get cost of minibatch

        cost = T.mean(L)

        #compute the gradients of the cost of the `dA` with respect to its params
        gparams = T.grad(cost,self.params)
        updates = [
            (param,param-learning_rate*gparams)
            for param,gparam in zip(self.params,gparams)
        ]

        return (cost,updates)


