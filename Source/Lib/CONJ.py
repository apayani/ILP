import numpy as np
import random
import collections
from time import sleep
from datetime import datetime
import sys
import tensorflow as tf
from itertools import product
from itertools import combinations_with_replacement
from .PredicateLibV5 import PredFunc

from .mylibw import *

class CONJ(PredFunc):
    def __init__(self,name='',trainable=True,sig=1.0,init=[-.1,.1],init_terms=[],predColl=None,off_w=-10,fast=False,neg=False):
        
        super().__init__(name,trainable)
        self.init_terms = init_terms
        self.mean_and,self.std_and = init
        self.col = [tf.GraphKeys.GLOBAL_VARIABLES,self.name]
        self.sig = sig
        self.predColl=predColl
        self.off_w=off_w
        self.fast=fast
        self.neg=neg


    def pred_func(self,xi,xcs=None,t=0):
        wa=None
        wo=None
        
        if not self.fast:
            if len(self.init_terms)>0:
                
                wa = np.zeros((1,self.predColl[self.name].Lx),dtype=np.float32) 
                
                for i,a in enumerate(self.init_terms):
                    if i>0:
                        raise ValueError('Only one term is allowed in conj')
                    for item in a.split(', '):
                        wa[i,self.predColl[self.name].get_term_index(item)]=10
                        
                
                wa[wa<1] =self.off_w
                

                
        else:

            if len(self.init_terms)>0:
                
                res=0.0
                for i,a in enumerate(self.init_terms):
                    if i>0:
                       raise ValueError('Only one term is allowed in conj')
                    resi=1
                    for item in a.split(', '):
                        
                        resi *= xi[:,self.predColl[self.name].get_term_index(item)]
                        
                    res = OR(res,resi)
                return tf.expand_dims(res,-1)
            
         
        res = logic_layer_and( xi, 1 ,  col=self.col, name=self.name+'_AND', sig=self.sig, mean=self.mean_and,std=self.std_and,w_init=wa)
        
        if self.neg:
            return 1.0-res
        return res
    
    def conv_weight_np(self,w):
        return sharp_sigmoid_np(w,self.sig)
    def conv_weight(self,w):
        return sharp_sigmoid(w,self.sig)
    def get_func(self,session,names,threshold=.2,print_th=True):

        wt = tf.get_collection(self.name)
        
        if len(wt)<1:
            return ''
        
        w_andt = wt[0]
        
        
        w_and = session.run( w_andt )
        w_and = sharp_sigmoid_np(w_and,self.sig)
        

        clause = []

        terms=[]
        k=0
        for v in range( w_and[k,:].size):
            if w_and[k,v]>threshold:
                if names is None:
                    terms.append( 'I_%d'%(v+1))
                else:
                    terms.append( names[v])

                if print_th and w_and[k,v]<.95:
                        terms[-1] = '[%.2f]'%(w_and[k,v]) + terms[-1]

        s = ','.join(terms)
        
        if self.neg:
            clause =  '\t :- not(' + s +' )'
        else:
            clause =  '\t :- (' + s +' )'

        return clause

    def get_item_contribution(self,session,names,threshold=.2 ):
         

        raise ValueError('Not Implemented')  