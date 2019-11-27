import tensorflow as tf
import numpy as np
import copy
import inspect

def OR(x,y):
    return x+y-x*y
#####################################################
def XOR(x,y):
    return x+y-2*x*y
#####################################################
def NOT(x):
    return 1.0-x
#####################################################
def mysplit(x,sz,axis=-1):
    return tf.split( x, num_or_size_splits=sz,axis=axis)
#####################################################
def update_dic( d , i , v, mode='max'):
    if mode=='max':
        if i in d:
            d[i] = max( d[i],v)
        else:
            d[i]=v
    if mode=='min':
        if i in d:
            d[i] = min( d[i],v)
        else:
            d[i]=v
    if mode=='add':
        if i in d:
            d[i] = d[i]+v
        else:
            d[i]=v
#####################################################
def read_by_tokens(fileobj):
    for line in fileobj:
        yield line.split() 
#####################################################
def prinT(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    print('********************')
    for var_name, var_val in callers_local_vars:
        if var_val is var:
            print( var_name, '=',var_val)
#####################################################
def partition_range( total_size, partition_size):
    res1=[]
    res2=[]
    i=0
    while i < total_size:
        end_range = min( total_size,i+partition_size)
        res1.append( (i,end_range))
        res2.append(end_range-i)
        i=end_range
    return res1,res2
#####################################################
def add_neg(x):
    return tf.concat((x,1.0-x),-1)
#####################################################
def myC(x,n=2):
    k=10**2
    return np.round(x*k)/k
#####################################################
def clip_grads_val( optimizer,loss,min_val,max_val,global_state=None):
    
    grad_vars = optimizer.compute_gradients(loss)
    print(grad_vars)
    clipped_gvs = [(tf.clip_by_value(grad, min_val, max_val), var) for grad, var in grad_vars]
    
    return optimizer.apply_gradients(clipped_gvs, global_step=global_state)
#####################################################
def custom_grad(fx,gx):
    t = gx
    return t + tf.stop_gradient(fx - t)
#####################################################
def FC(inputs,sizes,activations=None,name='fc'):
    if isinstance(inputs,(list,tuple)):
        inputs = tf.concat(inputs,-1)
    if not isinstance(sizes,(list,tuple)):
        sizes=[sizes]
    X = inputs
    if not isinstance(activations,(list,tuple)):
        activations = [activations] * len(sizes)

    for i in range(len(sizes)):
        if len(activations) < i-1:
            act = None
        else:
            act = activations[i]
        X = tf.layers.dense( X, sizes[i] , act , name=name+'_%d'%(i+1) )
         
    return X
#####################################################
def weight_variable(shape, stddev=0.01, name='weight'):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)
#####################################################
def bias_variable(shape, value=0.0, name='bias'):
    # print(sss,'shape=',shape)
    initial = tf.constant(value, shape=shape)
    return tf.Variable(initial, name=name)
#####################################################
def sig(x,p=1.):
    xc = np.clip(x,-30,30)
    return np.exp(xc*p)/(1.0+np.exp(xc*p))
#####################################################
def npsig(x,p=1.):
    xc = np.clip(x,-30,30)
    return np.exp(xc*p)/(1.0+np.exp(xc*p))
#####################################################
BSP=50
sss="-----------------------------------------------\n"
and_op = lambda x,ax=-1:   tf.reduce_prod( x  , axis=ax,name='my_prod') 
or_op = lambda x,ax=-1:  1.0-tf.reduce_prod( 1.0-x  , axis=ax) 
#####################################################
def make_batch(batch_size,v):
    return tf.tile( [v] , [batch_size,1])
#####################################################
def relu1(x):
        return tf.nn.relu ( 1.0 - tf.nn.relu(1.0-x) )
def leaky_relu1(x):
        return tf.nn.leaky_relu ( 1.0 - tf.nn.leaky_relu(1.0-x) )
#####################################################
def neg_ent_loss(label,prob,eps=1.0e-4):
    return - ( label*tf.log(eps+prob) + (1.0-label)*tf.log(eps+1.0-prob))
#####################################################
def neg_ent_loss_p(label,prob,p=.5,eps=1.0e-4):
    return - ( p*label*tf.log(eps+prob) + (1.0-p)*(1.0-label)*tf.log(eps+1.0-prob))
#####################################################
def sharp_sigmoid(x,c=5):
        cx = c*x
        return tf.sigmoid(cx)
        cx = tf.clip_by_value(cx,-30,30)
        return tf.sigmoid(cx)
#####################################################
def sharp_sigmoid_np(x,c=5):
        cx = c*x
        cx = np.clip(cx,-30,30)
        
        return 1./ (1+ np.exp( -cx ) )
#####################################################
def _concat(prefix, suffix, static=False):

    if isinstance(prefix, tf.Tensor):
        p = prefix
        p_static = tensor_util.constant_value(prefix)
        if p.shape.ndims == 0:
            p = tf.expand_dims(p, 0)
        elif p.shape.ndims != 1:
            raise ValueError("prefix tensor must be either a scalar or vector, "
                "but saw tensor: %s" % p)
    else:
        p = tensor_shape.as_shape(prefix)
        p_static = p.as_list() if p.ndims is not None else None
        p = (tf.constant(p.as_list(), dtype=dtypes.int32)
            if p.is_fully_defined() else None)

    if isinstance(suffix, tf.Tensor):
        s = suffix
        s_static = tensor_util.constant_value(suffix)
        if s.shape.ndims == 0:
            s = tf.expand_dims(s, 0)
        elif s.shape.ndims != 1:
            raise ValueError("suffix tensor must be either a scalar or vector, "
                    "but saw tensor: %s" % s)
    else:
        s = tensor_shape.as_shape(suffix)
        s_static = s.as_list() if s.ndims is not None else None
        s = (tf.constant(s.as_list(), dtype=tf.int32)
            if s.is_fully_defined() else None)

    if static:
        shape = tensor_shape.as_shape(p_static).concatenate(s_static)
        shape = shape.as_list() if shape.ndims is not None else None
    else:
        if p is None or s is None:
            raise ValueError("Provided a prefix or suffix of None: %s and %s"
                % (prefix, suffix))
        shape = tf.concat((p, s), 0)
    return shape
#####################################################
class RandomBinary(object):
   
    def __init__(self, k,s, seed=0, dtype=tf.float32):
        self.k=k
        self.s=s
        self.seed = seed
        self.dtype = tf.as_dtype(dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        
        if dtype is None:
            dtype = self.dtype
        
        
        if shape[0]==1 and len(shape)>2:
            inc=1
        else:
            inc = 0 
         

        logit = tf.ones([shape[inc], shape[inc+1]]) /shape[inc+1]
        v1 = tf.multinomial(logit, self.k)
        v2 = tf.one_hot( v1,shape[inc+1])
        v3=tf.reduce_sum(v2,axis=-2)
        v3 = tf.reshape( v3, shape)
        v=  5*(relu1(v3)/1.9-.5 )
        return v 

    def get_config(self):
        return {
            "alpha": self.alpha,
            "seed": self.seed,
            "dtype": self.dtype.name
        }
#####################################################
def logic_layer_and_old(inputs, units,scope=None, col=None ,name="W", trainable=True, sig =1.,mean=0.0,std=2.,w_init=None,rescale=False):
    if isinstance(inputs,tuple) or isinstance(inputs,list):
        inputs = tf.concat(inputs,axis=-1)

    V = inputs
    L = V.get_shape().as_list()[-1]
    
    
    if w_init is not None:
        init = tf.constant_initializer(w_init)
    else:
        if std<0: 
            init = RandomBinary(-std,.551)
        else:
            init = tf.truncated_normal_initializer(mean=mean,stddev=std)

    
    if scope is not None:
        with tf.variable_scope( scope, tf.AUTO_REUSE):
            W= tf.get_variable(name, [units, L ],initializer= init,regularizer=None,dtype=tf.float32,  collections=col, trainable= trainable)
    else:
        W= tf.get_variable(name , [units, L ],initializer= init,regularizer=None,dtype=tf.float32,  collections=col, trainable= trainable)
    

    if sig>0:
        W = sharp_sigmoid(W,sig)
    else:
        W = relu1(W)

    Z = tf.expand_dims(W,axis=0) * (1.0-tf.expand_dims(V,axis=1) )
    S=and_op( 1.0-Z  )

    return S
#####################################################
def logic_layer_or_old(inputs, units,scope=None, col=None ,name="W", trainable=True, sig =1.,mean=0.0,std=2.,w_init=None,rescale=False):
    
    if isinstance(inputs,tuple) or isinstance(inputs,list):
        inputs = tf.concat(inputs,axis=-1)

    V = inputs
    L = V.get_shape().as_list()[1]
    
    
    if w_init is not None:
        init = tf.constant_initializer(w_init)
    else:
        if std<0: 
            init = RandomBinary(-std,.551)
        else:
            init = tf.truncated_normal_initializer(mean=mean,stddev=std)

    
    if scope is not None:
        with tf.variable_scope( scope, tf.AUTO_REUSE):
            W= tf.get_variable(name, [units, L ],initializer= init,regularizer=None,dtype=tf.float32,  collections=col, trainable= trainable)
    else:
        W= tf.get_variable(name , [units, L ],initializer= init,regularizer=None,dtype=tf.float32,  collections=col, trainable= trainable)

    if sig>0:
        W = sharp_sigmoid(W,sig)
    else:
        W = relu1(W)

    Z = tf.expand_dims(W,axis=0) * tf.expand_dims(V,axis=1)
    S = 1.0-and_op( 1.0-Z )
     
    return S
#####################################################
def logic_layer_and(inputs, units,scope=None, col=None ,name="W", trainable=True, sig =1.,mean=0.0,std=2.,w_init=None,rescale=False):
    if isinstance(inputs,tuple) or isinstance(inputs,list):
        inputs = tf.concat(inputs,axis=-1)

    V = inputs
    L = V.get_shape().as_list()[-1]
    size = len(V.get_shape().as_list() )
    
    if w_init is not None:
        init = tf.constant_initializer(w_init)
    else:
        if std<0: 
            init = RandomBinary(-std,.551)
        else:
            init = tf.truncated_normal_initializer(mean=mean,stddev=std)

    
    
    if units==1:

        if scope is not None:
            
            with tf.variable_scope( scope, tf.AUTO_REUSE):
                W= tf.get_variable(name, [1,L ],initializer= init,regularizer=None,dtype=tf.float32,  collections=col, trainable= trainable)
        else:
            W= tf.get_variable(name , [1,L ],initializer= init,regularizer=None,dtype=tf.float32,  collections=col, trainable= trainable)
        

        if sig>0:
            W = sharp_sigmoid(W,sig)
        else:
            W = relu1(W)


        for _ in range(size-2):
            W = tf.expand_dims(W,axis=0) 


        Z = W * (1.0-V)
        S=and_op( 1.0-Z  )
        S=tf.expand_dims(S,-1)


    else:

        if scope is not None:
            with tf.variable_scope( scope, tf.AUTO_REUSE):
                W= tf.get_variable(name, [units, L ],initializer= init,regularizer=None,dtype=tf.float32,  collections=col, trainable= trainable)
        else:
            W= tf.get_variable(name , [units, L ],initializer= init,regularizer=None,dtype=tf.float32,  collections=col, trainable= trainable)
        

        if sig>0:
            W = sharp_sigmoid(W,sig)
        else:
            W = relu1(W)


        for _ in range(size-1):
            W = tf.expand_dims(W,axis=0) 


        Z = W * (1.0-tf.expand_dims(V,axis=-2) )
        S=and_op( 1.0-Z  )

    return S
def logic_layer_or(inputs, units,scope=None, col=None ,name="W", trainable=True, sig =1.,mean=0.0,std=2.,w_init=None,rescale=False):
    
    if isinstance(inputs,tuple) or isinstance(inputs,list):
        inputs = tf.concat(inputs,axis=-1)

    V = inputs
    L = V.get_shape().as_list()[-1]
    size = len(V.get_shape().as_list() )
    
    
    if w_init is not None:
        init = tf.constant_initializer(w_init)
    else:
        if std<0: 
            init = RandomBinary(-std,.551)
        else:
            init = tf.truncated_normal_initializer(mean=mean,stddev=std)

    
    if scope is not None:
        with tf.variable_scope( scope, tf.AUTO_REUSE):
            W= tf.get_variable(name, [units, L ],initializer= init,regularizer=None,dtype=tf.float32,  collections=col, trainable= trainable)
    else:
        W= tf.get_variable(name , [units, L ],initializer= init,regularizer=None,dtype=tf.float32,  collections=col, trainable= trainable)

    if sig>0:
        W = sharp_sigmoid(W,sig)
    else:
        W = relu1(W)

    
    if units==1:
    
        for _ in range(size-2):
            W = tf.expand_dims(W,axis=0) 
        
        # Z = W* V

        S = 1.0-tf.reduce_prod( 1- W*V , axis=-1,keep_dims=True) 
        # S = 1.0-and_op( 1.0-Z )
    
    else:

        for _ in range(size-1):
            W = tf.expand_dims(W,axis=0) 
        
        Z = W* tf.expand_dims(V,axis=-2)
        S = 1.0-and_op( 1.0-Z )
    
     
    return S
 

def logic_layer_and_multi(inputs, units,n1=10,n2=2,scope=None, col=None ,name="W",  sig =1.,mean=0.0,std=2. ):
    
    if isinstance(inputs,tuple) or isinstance(inputs,list):
        inputs = tf.concat(inputs,axis=-1)

    V = inputs
    L = V.get_shape().as_list()[-1]
    
    s =  0
    sz = [] 
    
    pad_size = (n1 -L%n1) %n1
    L_new = L + pad_size
    cnt = L_new//n1
    
    
    
    V_new = tf.pad( V, [ [0,0], [0,pad_size] ] )
    V_new = tf.reshape( V_new, [-1, 1,cnt , 1,n1])
     
    w_sm = tf.get_variable( name+'_SMX' , [1,units,cnt,n2,n1], initializer=tf.random_uniform_initializer(0.,.1),  collections=col)
    s = tf.reduce_sum( tf.nn.softmax(5*w_sm) * V_new, -1)
    s= tf.reshape( s, [-1,units,cnt*n2])

    LW = cnt*n2
    init = tf.truncated_normal_initializer(mean=mean,stddev=std)
    W= tf.get_variable(name+'_AND', [1, units,LW ],initializer= init,regularizer=None,dtype=tf.float32,  collections=col)
    
    if sig>0:
        W = sharp_sigmoid(W,sig)
    else:
        W = relu1(W)

    
    Z = W * (1.0-s)
    S=and_op( 1.0-Z)
    
    return S
#####################################################
def get_nb_params_shape(shape):
    '''
    Computes the total number of params for a given shap.
    Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
    '''
    nb_params = 1
    for dim in shape:
        nb_params = nb_params*int(dim)
    return nb_params 
#####################################################
def count_number_trainable_params():
    '''
    Counts the number of trainable variables.
    '''
    tot_nb_params = 0
    for trainable_variable in tf.trainable_variables():
        shape = trainable_variable.get_shape() # e.g [D,F] or [W,H,C]
        current_nb_params = get_nb_params_shape(shape)
        tot_nb_params = tot_nb_params + current_nb_params
    return tot_nb_params
#####################################################
 
  