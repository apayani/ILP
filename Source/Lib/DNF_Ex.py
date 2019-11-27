
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

class DNF_Ex(PredFunc):
    def __init__(self,name='',trainable=True,terms=4,sig=1.0,init=[.1,-1,.1,.1],N1=20,N2=2):
        
        super().__init__(name,trainable)
        self.terms = terms
        self.mean_or,self.std_or,self.mean_and,self.std_and = init
        self.col = [tf.GraphKeys.GLOBAL_VARIABLES,self.name]
        self.sig = sig
        self.N1=N1
        self.N2=N2


    def pred_func(self,xi,xcs=None,t=0):
        
        temp = logic_layer_and_multi( xi, self.terms , self.N1,self.N2, col=self.col,name=self.name+'_AND',sig=self.sig,mean=self.mean_and,std=self.std_and) 
        # temp = logic_layer_and( xi, self.terms ,  col=self.col, name=self.name+'_AND', sig=self.sig, mean=self.mean_and,std=self.std_and)
        res = logic_layer_or( temp, 1 ,  col=self.col, name=self.name+'_OR',  sig=self.sig, mean=self.mean_or,std=self.std_or ) 
        return res
    
    def get_func(self,session,names,threshold=.2,print_th=True):

        wt = tf.get_collection(self.name)
        
        if len(wt)<2:
            return ''
        
        for w in wt:
            if '_AND' in w.name:
                w_andt=w
            if '_OR' in w.name:
                w_ort=w
            if '_SMX' in w.name:
                w_smt=w
                
         

        w_and,w_or,w_sm = session.run( [w_andt,w_ort,w_smt] )
        
        w_and = sharp_sigmoid_np(w_and,self.sig)
        w_or = sharp_sigmoid_np(w_or,self.sig)
    

      

        def get_multi_name( sm,w,n1,n2,th=.5):
            pad_size = (n1 -len(names)%n1) %n1
            for i in range(pad_size):
                names.append('_placeholder_%d'%i)
            all_names=[]
            # if sig>w:
            #     w = npsig(w)
            for i in range(w.shape[1]):
                names_i = []
                for j in range(w.shape[-1]):
                    j1 = j//n2
                    j2= j - j1*n2

                    ind_sm = np.argmax(sm[0,i,j1,j2,:])
                    ind = j1*n1+ind_sm
                    names_i.append( names[ind] )
                all_names.append(names_i)
            return all_names


        
            all_names = get_multi_name( w_sm,w_and,self.args.N1,self.args.N2, names)
            for k in range(w_or[0,:].size):
                if w_or[0,k]>threshold:
                    s=""
                    # s = ' , '.join(all_names[k])
                    for v in range(w_and.shape[2]):
                        if w_and[0,k,v]>threshold:
                            if s=="" :
                                s=all_names[k][v]
                    #            s = self.IndexToName(v,pred,neg=pred.neg)
                               
                            else:
                                s = s+ ' , '+ all_names[k][v]
                            if print_th:
                                s+='[%.1f]'%(w_and[0,k,v])
                    if print_th:
                        print(pred_name+ '(' + ','.join(variable_list[0:pred.arity]) + ')[%.1f]'%(w_or[0,k])+ '  :-  ' + s)
                    else:
                        print(pred_name+ '(' + ','.join(variable_list[0:pred.arity]) + ')  :-  ' + s)

        all_names = get_multi_name( w_sm,w_and,self.N1,self.N2, names)
        clauses = []

        for k in range(w_or[0,:].size):
            if w_or[0,k]>threshold:
                
                terms=[]
                for v in range( w_and.shape[2]):
                    if w_and[0,k,v]>threshold:
                        if names is None:
                            terms.append( 'I_%d'%(v+1))
                        else:
                            terms.append( all_names[k][v])

                        if print_th and w_and[0,k,v]<.95:
                                terms[-1] = '[%.1f]'%(w_and[0,k,v]) + terms[-1]

                s = ','.join(terms)
                if print_th and w_or[0,k]<.95:
                    clauses.append( '\t :- [%.1f] ('%(w_or[0,k]) + s +' )')
                else:
                    clauses.append( '\t :- ('%(w_or[0,k]) + s +' )')
        return '\n'.join(clauses)



    def conv_weight_np(self,w):
        return sharp_sigmoid_np(w,self.sig)
    
    def conv_weight(self,w):
        return sharp_sigmoid(w,self.sig)
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
        # max_or=1.
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
                        #     items[tn] = max( items[tn], w_and[k,v] )
                        # else:
                        #     items[tn] =   w_and[k,v] 
                        if tn in items:
                            items[tn] = max( items[tn],(w_or[0,k] * w_and[k,v] /max_or)/max_and  )
                        else:
                            items[tn] = (w_or[0,k] * w_and[k,v] /max_or)/max_and

        return items   