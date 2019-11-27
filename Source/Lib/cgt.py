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

class CGT(PredFunc):
    def __init__(self,name='',trainable=True,predColl=None,init_w=0.0,c=10.0):
        
        super().__init__(name,trainable)
        self.predColl = predColl
        self.w=None
        self.init_w=init_w
        self.c=c
    def pred_func(self,xi,xcs,t):

        # print(self.name,xi.shape)
        self.w = bias_variable(shape=(),value=self.init_w) 
        return sharp_sigmoid( xi - self.w ,self.c)
        # return xi[:,0]
     
    def get_func(self,session,names,threshold=.2,print_th=True):
        wv = session.run( self.w)
        return '>%.1f'%wv