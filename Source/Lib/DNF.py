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

class DNF(PredFunc):
    def __init__(self,name='',trainable=True,terms=4,sig=1.0,init=[.1,-1,.1,.1],init_terms=[],post_terms=[],predColl=None,off_w=-10,fast=False,neg=False):
        
        super().__init__(name,trainable)
        self.terms = terms
        self.init_terms = init_terms
        self.post_terms = post_terms
        self.mean_or,self.std_or,self.mean_and,self.std_and = init
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
                
                wa = np.zeros((self.terms,self.predColl[self.name].Lx),dtype=np.float32) 
                wo = np.zeros((1,self.terms)) 
                for i,a in enumerate(self.init_terms):
                    
                    for item in a.split(', '):
                        wa[i,self.predColl[self.name].get_term_index(item)]=10
                        wo[0,i]=10
                
                wa[wa<1] =self.off_w
                wo[wo<1] =self.off_w

                
        else:

            if len(self.init_terms)>0:
                
                res=0.0
                for i,a in enumerate(self.init_terms):
                    
                    resi=1
                    for item in a.split(', '):
                        
                        resi *= xi[:,self.predColl[self.name].get_term_index(item)]
                        
                    res = OR(res,resi)
                return tf.expand_dims(res,-1)
            
         
        temp = logic_layer_and( xi, self.terms ,  col=self.col, name=self.name+'_AND', sig=self.sig, mean=self.mean_and,std=self.std_and,w_init=wa,trainable=self.trainable)
        res = logic_layer_or( temp, 1 ,  col=self.col, name=self.name+'_OR',  sig=self.sig, mean=self.mean_or,std=self.std_or,w_init=wo,trainable=self.trainable ) 

        if self.neg:
            res=1.0-res
            
        for t in self.post_terms:
            ind = self.predColl[self.name].get_term_index(t[1])
            if t[0]=='and':
                res = res * xi[:,ind:(ind+1)]
            if t[0]=='or':
                res = 1.0- (1.0-res) * (1.0-xi[:,ind:(ind+1)] )
                # res=logic_layer_or( (res,xi[:,ind:(ind+1)]), 1 ,  col=self.col, name=self.name+'_xx',  sig=self.sig, mean=self.mean_or,std=self.std_or ) 
        
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
                if print_th and w_or[0,k]<.9999:
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
                max_and=1
                for v in range( w_and[k,:].size):
                    if w_and[k,v]>threshold:
                        if names is None:
                            tn = 'I_%d'%(v+1)
                        else:
                            tn=  names[v]

                        # if tn in items:
                        #     items[tn] = max( items[tn], w_and[k,v] )
                        # else:
                        #     items[tn] =   w_and[k,v] 
                        if tn in items:
                            items[tn] = max( items[tn],(w_or[0,k] * w_and[k,v] /max_or)/max_and  )
                        else:
                            items[tn] = (w_or[0,k] * w_and[k,v] /max_or)/max_and

        return items   