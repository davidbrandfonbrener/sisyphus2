#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:54:58 2017

@author: daniel
"""

import numpy as np


class weight_initializer(object):
    
    def __init__(self,params,init_weights_path):
        self.N_in       = params['N_in']
        self.N_rec      = params['N_rec']
        self.N_out      = params['N_out']
        self.init_weights_path = init_weights_path
        self.init_type = params['init_type']

    def gen_weight_dict(self):
        
        if self.init_type == 'gauss':
            weights_path = self.gaussian_spec_rad()
        elif self.init_type == 'identity':
            weights_path = self.alpha_Identity()
        elif self.init_type == 'feed_forward':
            weights_path = self.feed_forward()
        elif self.init_type == 'strict_feed_forward':
            weights_path = self.strict_feed_forward()
            
        return weights_path
            

    def gaussian_spec_rad(self,spec_rad=1.1):
        '''Generate random gaussian weights with specified spectral radius'''
        N_in    = self.N_in
        N_rec   = self.N_rec
        N_out   = self.N_out
        
        weights_path = self.init_weights_path
    
        #Uniform between -.1 and .1
        W_in = .2*np.random.rand(N_rec,N_in) - .1
        W_out = .2*np.random.rand(N_out,N_rec) - .1
        
        b_rec = np.zeros(N_rec)
        b_out = np.zeros(N_out)
        
        init_state = .1 + .01*np.random.randn(N_rec)
        
        W_rec = np.random.randn(N_rec,N_rec)
        W_rec = 2.*W_rec/np.max(np.abs(np.linalg.eig(W_rec)[0]))
        
        input_Connectivity = np.ones([N_rec,N_in])
        rec_Connectivity = np.ones([N_rec,N_rec])
        output_Connectivity = np.ones([N_out,N_rec])
        
        np.savez(weights_path, W_in = W_in,
                                W_rec = W_rec,
                                W_out = W_out,
                                b_rec = b_rec,
                                b_out = b_out,
                                init_state = init_state,
                                input_Connectivity = input_Connectivity,
                                rec_Connectivity= rec_Connectivity,
                                output_Connectivity=output_Connectivity)
        
        return weights_path
        
    def alpha_Identity(self,alpha=1.0):
        '''Generate recurrent weights w(i,i) = alpha, w(i,j) = 0'''
       
        N_in    = self.N_in
        N_rec   = self.N_rec
        N_out   = self.N_out
        
        weights_path = self.init_weights_path
    
        #Uniform between -.1 and .1
        W_in = .2*np.random.rand(N_rec,N_in) - .1
        W_out = .2*np.random.rand(N_out,N_rec) - .1
        
        b_rec = np.zeros(N_rec)
        b_out = np.zeros(N_out)
        
        init_state = .1 + .01*np.random.randn(N_rec)
        
        W_rec = np.eye(N_rec)*alpha
        
        input_Connectivity = np.ones([N_rec,N_in])
        rec_Connectivity = np.ones([N_rec,N_rec])
        output_Connectivity = np.ones([N_out,N_rec])
        
        np.savez(weights_path, W_in = W_in,
                                W_rec = W_rec,
                                W_out = W_out,
                                b_rec = b_rec,
                                b_out = b_out,
                                init_state = init_state,
                                input_Connectivity = input_Connectivity,
                                rec_Connectivity= rec_Connectivity,
                                output_Connectivity=output_Connectivity)
        
        return weights_path
        
    def feed_forward(self,sigma=.1):
        '''Generate random feedforward wrec (lower triangular)'''
       
        N_in    = self.N_in
        N_rec   = self.N_rec
        N_out   = self.N_out
        
        weights_path = self.init_weights_path
    
        #Uniform between -.1 and .1
        W_in = .2*np.random.rand(N_rec,N_in) - .1
        W_out = .2*np.random.rand(N_out,N_rec) - .1
        
        b_rec = np.zeros(N_rec)
        b_out = np.zeros(N_out)
        
        init_state = .1 + .01*np.random.randn(N_rec)
        
        W_rec = np.tril(sigma*np.random.randn(N_rec,N_rec),-1)
        
        input_Connectivity = np.ones([N_rec,N_in])
        rec_Connectivity = np.ones([N_rec,N_rec])
        output_Connectivity = np.ones([N_out,N_rec])
        
        np.savez(weights_path, W_in = W_in,
                                W_rec = W_rec,
                                W_out = W_out,
                                b_rec = b_rec,
                                b_out = b_out,
                                init_state = init_state,
                                input_Connectivity = input_Connectivity,
                                rec_Connectivity= rec_Connectivity,
                                output_Connectivity=output_Connectivity)
        
        return weights_path
        
    def strict_feed_forward(self,alpha=1.):
        '''Generate random feedforward wrec (lower triangular)'''
       
        N_in    = self.N_in
        N_rec   = self.N_rec
        N_out   = self.N_out
        
        weights_path = self.init_weights_path
    
        #Uniform between -.1 and .1
        W_in = .2*np.random.rand(N_rec,N_in) - .1
        W_out = .2*np.random.rand(N_out,N_rec) - .1
        
        b_rec = np.zeros(N_rec)
        b_out = np.zeros(N_out)
        
        init_state = .1 + .01*np.random.randn(N_rec)
        
        W_rec = np.zeros([N_rec,N_rec])
        for ii in range(N_rec-1):
            W_rec[ii+1,ii] = alpha
        
        input_Connectivity = np.ones([N_rec,N_in])
        rec_Connectivity = np.ones([N_rec,N_rec])
        output_Connectivity = np.ones([N_out,N_rec])
        
        np.savez(weights_path, W_in = W_in,
                                W_rec = W_rec,
                                W_out = W_out,
                                b_rec = b_rec,
                                b_out = b_out,
                                init_state = init_state,
                                input_Connectivity = input_Connectivity,
                                rec_Connectivity= rec_Connectivity,
                                output_Connectivity=output_Connectivity)
        
        return weights_path

    