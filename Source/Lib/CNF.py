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

class CNF(PredFunc):
    def __init__(self,name='',trainable=True,terms=4,sig=1.0,init=[.1,-1,.1,.1]):
        
        super().__init__(name,trainable)
        self.terms = terms
        self.mean_or,self.std_or,self.mean_and,self.std_and = init
        self.col = [tf.GraphKeys.GLOBAL_VARIABLES,self.name]
        self.sig = sig

    def pred_func(self,xi,xcs=None,t=0):
        
        
        temp = logic_layer_or( xi, self.terms ,  col=self.col, name=self.name+'_AND', sig=self.sig, mean=self.mean_and,std=self.std_and)
        res = logic_layer_and( temp, 1 ,  col=self.col, name=self.name+'_OR',  sig=self.sig, mean=self.mean_or,std=self.std_or ) 
        return res
    
    def conv_weight_np(self,w):
        return sharp_sigmoid_np(w,self.sig)
    
    def conv_weight(self,w):
        return sharp_sigmoid(w,self.sig)
    def get_func(self,session,names,threshold=.2,print_th=True):

        wt = tf.get_collection(self.name)
        
        if len(wt)<2:
            return ''
        
        if '_AND' in wt[0].name:
            w_andt = wt[0]
            w_ort  = wt[1]
        else:
            w_andt = wt[1]
            w_ort  = wt[0]

        w_and,w_or = session.run( [w_andt,w_ort] )
        w_and = sharp_sigmoid_np(w_and,self.sig)
        w_or = sharp_sigmoid_np(w_or,self.sig)
    

        clauses = []

        for k in range(w_or[0,:].size):
            if w_or[0,k]>threshold:
                
                terms=[]
                for v in range( w_and[k,:].size):
                    if w_and[k,v]>threshold:
                        if names is None:
                            terms.append( 'I_%d'%(v+1))
                        else:
                            terms.append( names[v])

                        if print_th and w_and[k,v]<.95:
                                terms[-1] = '[%.2f]'%(w_and[k,v]) + terms[-1]

                s = ','.join(terms)
                if print_th and w_or[0,k]<.95:
                    clauses.append( '\t :- [%.2f] ('%(w_or[0,k]) + s +' )')
                else:
                    clauses.append( '\t :- ('%(w_or[0,k]) + s +' )')
        return '\n'.join(clauses)
    
    def get_item_contribution(self,session,names,threshold=.2 ):
        items = {}

        wt = tf.get_collection(self.name)
        
        if len(wt)<2:
            return ''
        
        if '_AND' in wt[0].name:
            w_andt = wt[0]
            w_ort  = wt[1]
        else:
            w_andt = wt[1]
            w_ort  = wt[0]

        w_and,w_or = session.run( [w_andt,w_ort] )
        w_and = sharp_sigmoid_np(w_and,self.sig)
        w_or = sharp_sigmoid_np(w_or,self.sig)
    

         
        max_or = np.max(w_or[0,:]) + 1e-3
        max_or=1.
        for k in range(w_or[0,:].size):
            if w_or[0,k]>threshold:
                max_and = np.max(w_and[k,:]) + 1e-3
                for v in range( w_and[k,:].size):
                    if w_and[k,v]>threshold:
                        if names is None:
                            tn = 'I_%d'%(v+1)
                        else:
                            tn=  names[v]

                        # if tn in items:
                        #     items[tn] = max( items[tn],w_or[0,k] * w_and[k,v] /max_or)
                        # else:
                        #     items[tn] = w_or[0,k] * w_and[k,v] /max_or

                        # if tn in items:
                        #     items[tn] = max( items[tn], w_and[k,v] )
                        # else:
                        #     items[tn] =   w_and[k,v] 
  
                        if tn in items:
                            items[tn] = max( items[tn],(w_or[0,k] * w_and[k,v] /max_or)/max_and  )
                        else:
                            items[tn] = (w_or[0,k] * w_and[k,v] /max_or)/max_and

        return items   