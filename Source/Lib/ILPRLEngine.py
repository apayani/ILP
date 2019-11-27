import numpy as np
import random
import collections
from time import sleep
from datetime import datetime
import sys
import tensorflow as tf
import os.path
from .mylibw import *
from itertools import product
from itertools import combinations_with_replacement
from .PredicateLibV5 import *
import inspect

def lineno():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno

class ILPRLEngine(object):

    def __init__(self,args,predColl,bgs,disp_fn=None ):
        print( 'Tensorflow Version : ', tf.__version__)
        
        self.args=args
        self.predColl = predColl 
        self.predColl.args=args
        self.bgs=bgs
        self.disp_fn=disp_fn
        tf.set_random_seed(self.args.SEED) 
        config=tf.ConfigProto( device_count = {'GPU': self.args.GPU} )
        config.allow_soft_placement = True
        config.log_device_placement = False
        self.session= tf.Session(config=config)

        self.plogent =  tf.placeholder("float32", [],name='plogent') 
        self.index_ins=OrderedDict({})
        self.X0 = OrderedDict({})
        self.target_mask = OrderedDict({})
        self.target_data = OrderedDict({})

        self.last_outp=None
        self.last_bgs=None
        self.mean_cost = None

        for p in self.predColl.preds:
           
            self.index_ins[p.dname] = tf.constant( self.predColl.InputIndices[p.dname])
            if p.oname not in self.X0:
                self.X0[p.oname] =          tf.placeholder("float32", [None,self.predColl.pairs_len[p.oname]] , name='input_x_' + p.oname)
                if p.pFunc is not None:
                    self.target_data[p.oname] = tf.placeholder("float32", [None,self.predColl.pairs_len[p.oname]] , name='target_data_' + p.oname)
                    self.target_mask[p.oname] = tf.placeholder("float32", [None,self.predColl.pairs_len[p.oname]] , name='target_mask_' + p.oname)

        
        
        self.define_model()
        print("summary all variables")
        
        for k in tf.trainable_variables():
            if( isinstance(k, tf.Variable) and len(k.get_shape().as_list())>1 ):
                print( str(k))
                if( self.args.TB==1):
                    tf.summary.histogram( k.name, k)
                    if len(k.get_shape().as_list())==2:
                        tf.summary.image(k.name, tf.expand_dims( tf.expand_dims(k,axis=0),axis=3))
                    if len(k.get_shape().as_list())==3:
                        tf.summary.image(k.name, tf.expand_dims( k,axis=3))

        if self.args.TB==1:
            self.all_summaries = tf.summary.merge_all()
    ############################################################################################
    def check_weights( self,sess , w_filt  ):
        
        for p in self.predColl.preds:
            wts = tf.get_collection( p.dname)
            if len(wts)==0:
                continue
            wvs = sess.run( wts )

            for wt,wv in zip(wts,wvs):
                if not wt.name.endswith(w_filt) :
                    continue
                wv_sig = p.pFunc.conv_weight_np(wv)
                
                sumneg = np.sum( np.logical_and(wv_sig>.1,wv_sig<.9))
                if sumneg > 0: 
                    print( "weights in %s are not converged yet :  %f"%(wt.name,sumneg))
                    return False
        return True
    ############################################################################################
    def filter_predicates( self,sess , w_filt,th=.5  ):
        
        old_cost,_=self.runTSteps(sess)
        for p in self.predColl.preds:
            wts = tf.get_collection( p.dname)
            if len(wts)==0:
                continue
            wvs = sess.run( wts )

            for wt,wv in zip(wts,wvs):
                if not wt.name.endswith(w_filt) :
                    continue
                
                
                wv_sig = p.pFunc.conv_weight_np(wv)
                for ind,val in  np.ndenumerate(wv_sig):
                    
                    if val>.5:
                        wv_backup = wv*1.0
                        wv[ind]=-20
                        sess.run( wt.assign(wv))
                        cost,_=self.runTSteps(sess)
                         
                        if cost-old_cost >th :
                            wv = wv_backup*1.0
                        else:
                            old_cost=cost
                            print( 'removing',wt,ind)
                            
                sess.run( wt.assign(wv))
    ############################################################################################
    def filter_predicates2( self,sess ,  th=.5  ):
        
        old_cost,_=self.runTSteps(sess)
        for p in self.predColl.preds:
            wts = tf.get_collection( p.name)
            if len(wts)==0:
                continue
            wvs = sess.run( wts )
            
            wand = None
            wor = None
            if 'AND' in wts[0].name:
                wand = wvs[0]
                wor = wvs[1]
                wandt = wts[0]
                wort = wts[1] 
            else:
                wand = wvs[1]
                wor = wvs[0]
                wandt = wts[1]
                wort = wts[0]
            
            wand_bk=wand*1.0
            wor_bk=wand*1.0

            wand_sig = p.pFunc.conv_weight_np(wand)
            wor_sig = p.pFunc.conv_weight_np(wor)

            
             
            for k in range(wor_sig[0,:].size):
                if wor_sig[0,k]>.1:
                    
                    wor[0,k]=-20
                    sess.run( wort.assign(wor))
                    cost,_=self.runTSteps(sess)
                    
                    if abs(cost-old_cost) >.1 :
                        wor[0,k] = wor_bk[0,k]
                    else:
                        old_cost=cost
                        print( 'removing',wort,k)
                        continue

                    for v in range( wand_sig[k,:].size):
                        if wand_sig[k,v]>.1:
                            
                            wand[k,v]=-20
                            sess.run( wandt.assign(wand))
                            cost,_=self.runTSteps(sess)
                            
                            if abs(cost-old_cost) >.1 :
                                wand[k,v] = wand_bk[k,v]
                            else:
                                old_cost=cost
                                print( 'removing',wandt,v)
                                continue


            sess.run( wort.assign(wor))
            sess.run( wandt.assign(wand))
    ############################################################################################
    def get_sensitivity_factor( self,sess , p ,target_pred ):
        
        target_data = self.SARG[self.target_data[target_pred.name]]
        target_mask  = self.SARG[self.target_mask[target_pred.name]]

        def getval():
            val =  sess.run( self.XOs[target_pred.name]  , self.SARG )
            err = np.sum(  (val-target_data)*target_mask )
            return err

        
        
        # old_cost,_=self.runTSteps(sess)
        old_cost = getval()


        factors = dict({})
        wts = tf.get_collection( p.name)
        if len(wts)==0:
            return factors
        wvs = sess.run( wts )

        
        for wt,wv in zip(wts,wvs):
             
            if 'AND' not in wt.name:
                continue
            
            wv_sig = p.pFunc.conv_weight_np(wv)
            wv_backup = wv*1.0

            wv = wv_backup*1.0
            wv[:]=-20 
            sess.run( wt.assign(wv))
            cost_all = getval()
            cost_all_diff = abs( cost_all-old_cost)+1e-3

            
            # print('val',val)
            for k in  range(wv_sig[0,:].size):
                
                if np.max( wv_sig[:,k] ) <.1:
                    continue
                wv = wv_backup*1.0
                
                wv[:,k]=-20 
                sess.run( wt.assign(wv))
                # cost,_=self.runTSteps(sess)
                cost=getval()    
                if abs(cost-old_cost) >1 :
                    sens=1.0
                else:
                    sens = (1e-3+abs(cost-old_cost) )/cost_all_diff

                if k<=len(p.inp_list):
                    item = p.inp_list[ k]
                    factors[item]=sens

            return factors
    ############################################################################################
    def get_sensitivity_factor1( self,sess , p ,target_pred ):
        
        target_data = self.SARG[self.target_data[target_pred.name]]
        target_mask  = self.SARG[self.target_mask[target_pred.name]]

        def getval():
            val =  sess.run( self.XOs[target_pred.name]  , self.SARG )
            err = np.sum(  (val-target_data)*target_mask )
            return err

        
        
        # old_cost,_=self.runTSteps(sess)
        old_cost = getval()


        factors = dict({})
        wts = tf.get_collection( p.name)
        if len(wts)==0:
            return factors
        wvs = sess.run( wts )

        
        for wt,wv in zip(wts,wvs):
             
            if 'AND' not in wt.name:
                continue
            
            wv_sig = p.pFunc.conv_weight_np(wv)
            wv_backup = wv*1.0

            wv = wv_backup*1.0
            wv[:]=-20
            sess.run( wt.assign(wv))
            cost_all = getval()
            cost_all_diff = abs( cost_all-old_cost)+1e-3

            
            # print('val',val)
            for ind,val in  np.ndenumerate(wv_sig):
                
                if val<.1:
                    continue
                wv = wv_backup*1.0
                
                wv[ind]=-20
                sess.run( wt.assign(wv))
                # cost,_=self.runTSteps(sess)
                cost=getval()    
                if abs(cost-old_cost) >1 :
                    sens=1.0
                else:
                    sens = (1e-3+abs(cost-old_cost) )/cost_all_diff

                if ind[-1]<=len(p.inp_list):
                    item = p.inp_list[ ind[-1]]
                    if item in factors:
                        factors[item] = max( factors[item], sens)
                    else:
                        factors[item]=sens

            return factors
    ############################################################################################
    def binarize( self,sess   ):
        for p in self.predColl.preds:
            wts = tf.get_collection( p.dname)
            if len(wts)==0:
                continue
            wvs = sess.run( wts )
            for wt,wv in zip(wts,wvs):
                wv = wv*1.6
                s = 20
                wv [ wv>s] =s
                wv[wv<-s] = -s
                sess.run( wt.assign(wv))
    ############################################################################################
    def define_model(self):
        
       
        XOs , L3 = self.getTSteps(self.X0 )
                
                
        L1=0
        L2=0
        
        for p in  self.predColl.preds:
            vs = tf.get_collection( p.dname)
            for wi in vs:
                if '_AND' in wi.name:
                    wi = p.pFunc.conv_weight(wi)
                    
                    L2 += tf.reduce_mean( wi*(1.0-wi))

                    s = tf.reduce_sum( wi,-1)
                    L1 += tf.reduce_mean(  tf.nn.relu( s-self.args.MAXTERMS)  )
                    # L1 += tf.reduce_mean(   s   )
        

        self.XOs=XOs
        self.loss_gr = tf.constant(0.,tf.float32)
        self.loss = tf.constant(0.,tf.float32)

         
        for p in self.predColl.preds:
           
            if p.pFunc is None:
                continue

            if  p not in self.predColl.outpreds:
                continue


            if self.args.L2LOSS==3:
                err = ( self.target_data[p.name] - XOs[p.name] ) * self.target_mask[p.name]
                err = tf.square(err)
                err = tf.nn.relu( err - .03)
                self.loss_gr +=   tf.reduce_mean(err,-1)  
            

            if self.args.L2LOSS==2:
                # XOs[p.name] = sharp_sigmoid(XOs[p.name]-.4,3)
                err = 2*neg_ent_loss_p (self.target_data[p.oname] , XOs[p.oname], self.args.PLOGENT ) * self.target_mask[p.oname]
                self.loss_gr +=  tf.reduce_mean(err,-1)  
            

            if self.args.L2LOSS==1:
                err = ( self.target_data[p.oname] - XOs[p.oname] ) * self.target_mask[p.oname]
                err = tf.square(err)
                self.loss_gr +=   tf.reduce_mean(err,-1)  
            
            if self.args.L2LOSS==0:
                err = neg_ent_loss (self.target_data[p.oname] , XOs[p.oname] ) * self.target_mask[p.oname]
                self.loss_gr +=  tf.reduce_mean(err,-1)  
            
           
        
            loss =  neg_ent_loss (self.target_data[p.oname] , XOs[p.oname] ) * self.target_mask[p.oname]
            self.loss += tf.reduce_sum ( loss )  
        

 

        if self.args.L1>0 or self.args.L2>0 or self.args.L3>0:
            self.loss_gr  += ( self.args.L1*L1 + self.args.L2*L2+self.args.L3*L3 )
        
        
        self.lastlog=10
        self.cnt=0
        self.counter=0
        self.SARG=None
        self.last_cost=None
    ############################################################################################
    # execute t step forward chain 
    def getTSteps(self,_X0):
        
        XOs = OrderedDict( _X0 )
        L3=0
        
        
        
        for t in range(self.args.T):

            x = tf.concat( list( XOs.values() ), -1)
            for p in self.predColl.preds:
                
                if t>=p.max_T:
                    continue
                # if t==0 and p.max_T!=1:
                #     continue
                # # if p.dname=="CNT":
                #     lenp = len(p.pairs)
                #     px = np.zeros( (self.args.BS,lenp) , np.float)
                #     if t<lenp:
                #         px[:,t]=1
                #     else:
                #         px[:,-1]=1
                    
                #     XOs[p.name] = tf.constant(px,tf.float32)
                #     continue

                if p.pFunc is None:
                    continue


              
                     
                
                # if self.args.SYNC==0:
                

                xis=[]
                for pp in self.predColl.outpreds:
                    if p.Lx_details_dic[pp.oname]==0:
                        continue
                    xis.append(XOs[pp.oname])
                
                x = tf.concat( xis, -1)
                # x = tf.concat( list( XOs.values() ), -1)

                xi=tf.gather( tf.pad( x, [[0,0],[1,0]], mode='CONSTANT', constant_values=0.0 )  ,self.index_ins[p.dname],axis=1  )
                s = xi.shape.as_list()[1]*xi.shape.as_list()[2]
               
      
                    
                self.xi = xi
                if p.Lx>0:
                    xi = tf.reshape( xi, [-1,p.Lx])
                    if p.use_neg:
                        xi = tf.concat( (xi,1.0-xi) ,-1)
                
                
                
                l = xi.shape.as_list()[0]
                
                if p.chunk_count>0:
                    xis = mysplit( xi, sz=p.chunk_count, axis=0)
                    xos=[]
                    for xi in xis:
                        with tf.variable_scope( "ILP", reuse=tf.AUTO_REUSE, auxiliary_name_scope=False):
                            xos.append( p.pFunc.pred_func(xi,None,t) )
                    
                    xi = tf.concat(xos,0)
                else:
                    with tf.variable_scope( "ILP", reuse=tf.AUTO_REUSE, auxiliary_name_scope=False):
                        xi = p.pFunc.pred_func(xi,None,t)


                xi = tf.reshape( xi  , [-1,]+self.index_ins[p.dname].shape.as_list()[:2] )
                 
                xi =  1.0-and_op( 1.0-xi,-1) 
                
                L3+=  tf.reduce_max( xi*(1.0-xi))
                
                # Amalgamate
                if p.Fam =='and':
                    XOs[p.oname] = XOs[p.oname] *  tf.identity(xi) 
                if p.Fam =='eq':
                    XOs[p.oname] = tf.identity(xi)
                if p.Fam =='or':
                    XOs[p.oname] = 1.0 - (1.0-XOs[p.oname] )*  (1.0-tf.identity(xi) )
                
                if p.Fam =='max':
                    XOs[p.oname] =tf.maximum( XOs[p.oname] , xi ) 
                
            
            

        return XOs,L3
    def runTSteps(self,session,is_train=False,it=-1):
         
        
        # if self.SARG is None:
        self.SARG = dict({})
        bgs = self.bgs(it,is_train)
        self.SARG[self.plogent] =self.args.PLOGENT
    
        used_pred=[]
        for p in self.predColl.outpreds:
    
            self.SARG[self.X0[p.oname]] = np.stack( [bg.get_X0(p.oname) for bg in bgs] , 0 )
            if p.pFunc is None:
                continue

            
            try:
                if self.args.RATIO_POS>0:
                    ratio = self.args.RATIO_POS
            except:
                 ratio=0
            if ratio == 0 or not is_train:
                self.SARG[self.target_data[p.oname]] = np.stack( [ bg.get_target_data(p.oname) for bg in bgs] , 0 )
                self.SARG[self.target_mask[p.oname]] = np.stack( [ bg.get_target_mask(p.oname) for bg in bgs] , 0 )
            else:
                data= np.stack( [ bg.get_target_data(p.oname) for bg in bgs] , 0 )
                mask=[]
                for bg in bgs:
                    d = bg.get_target_data(p.oname)
                    m = bg.get_target_mask(p.oname) 
                    n_pos = np.sum( d*m )
                    n_neg = int( n_pos * ratio )
                    k = np.argwhere( m*(1-d)).flatten()
                    inds = np.random.permutation( k.size )
                    if n_neg<k.size:
                        m[ k [ inds[n_neg:]] ] = 0 
                    mask.append(m)

                mask = np.stack(mask,0)
                self.SARG[self.target_data[p.oname]] = data
                self.SARG[self.target_mask[p.oname]] = mask




            
        
      
          
        self.SARG[self.LR] = .001
        try:
            if is_train:
                if bool(self.args.LR_SC) :
                    for l,r in self.args.LR_SC:
                        if self.lastlog >= l and self.lastlog < r:
                            self.SARG[self.LR] = self.args.LR_SC[(l,r)]
                            break

        except:
            self.SARG[self.LR] = .001
        
        
         
        
        
        
        if is_train:
            _,cost,outp =  session.run( [self.train_op,self.loss,self.XOs ] , self.SARG)
        else:
            cost,outp =  session.run( [self.loss,self.XOs  ] , self.SARG ) 
        
        if is_train:
            self.last_outp=outp
            self.last_bg=bgs
            self.last_cost=cost
        try:
            self.lastlog = cost
        except:
            pass
        return cost,outp
    ############################################################################################
    def train_model(self):
        
        session  = self.session
        
        t1 =  datetime.now()
        print ('building optimizer...')
        self.LR = tf.placeholder("float", shape=(),name='learningRate')
        
        
        loss = tf.reduce_mean(self.loss_gr)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.LR, beta1=self.args.BETA1, 
                       beta2=self.args.BETA2, epsilon=self.args.EPS, 
                       use_locking=False, name='Adam')
         
        self.train_op  = self.optimizer.minimize(loss)
        
       
        
        t2=  datetime.now()
        print ('building optimizer finished. elapsed:' ,str(t2-t1))

        init = tf.global_variables_initializer()
        session.run(init)


        if self.args.TB==1:
            train_writer = tf.summary.FileWriter(self.args.LOGDIR, session.graph)
            train_writer.close()

        print( '***********************')
        print( 'number of trainable parameters : {}'.format(count_number_trainable_params()))
        print( '***********************')

        
        start_time = datetime.now()
         
        
        sum_cost = 0
        sum_cost_counter=0
        for i in range(self.args.ITER):
            

            cost,outp= self.runTSteps(session,True,i)
            
            sum_cost+=cost
            sum_cost_counter+=1
            if i % self.args.ITER2 == 0 and not np.isnan(np.mean(cost)):
                
                
                self.mean_cost = sum_cost/sum_cost_counter 
                sum_cost_counter=0
                sum_cost=0

                cost,outp= self.runTSteps(session,False,i)
                
                if self.disp_fn is not None:
                    self.disp_fn(self, i//self.args.ITER2,session,cost,outp)
                
                now = datetime.now()
                print('------------------------------------------------------------------')
                errs=OrderedDict({})
                for p in self.predColl.outpreds:
                    if p.pFunc is None:
                        continue
                    if np.sum(self.SARG[self.target_mask[p.oname]]) >0:
                        errs[p.oname] = np.sum (  (np.abs(outp[p.oname]-self.SARG[self.target_data[p.oname]] )) * self.SARG[self.target_mask[p.oname]] )
        
                print( 'epoch=' ,i//self.args.ITER2 , 'training cost=', self.mean_cost,'testing cost=', cost,  'elapsed : ',  str(now-start_time)  )# ,'mismatch counts', errs)
                names=[]
                
                #displaying outputs ( value vectors)
                for bs in self.args.DISP_BATCH_VALUES:
                    if bs>0:
                        break
                    cnt = 0
                    
                    for p in self.predColl.outpreds:
                        print_names=[]
                        if p.pFunc is None:
                            continue
                        

                        mask = self.SARG[self.target_mask[p.oname]]
                        target = self.SARG[self.target_data[p.oname]]
                        if np.sum(mask) >0:
                            
                            for ii in range( p.pairs_len ):
                                if mask[bs,ii]==1:
                                    if cnt<self.args.MAX_DISP_ITEMS:
                                        print_names.append(  '[('+ ','.join( p.pairs[ii]) +')],[%2.01f,%d]  '%(outp[p.oname][ bs,ii],target[bs,ii]))
                                        if  abs(outp[p.oname][bs,ii]-target[bs,ii]) >.3:
                                            print_names[-1] = '*'+print_names[-1]
                                        else:
                                            print_names[-1] = ' '+print_names[-1]
                                            
                                            
                                        if  cnt%10==0:
                                            print_names[-1] = '\n' +print_names[-1]
                                        cnt+=1
                                    
                                    else:
                                        break
                        print( ' , '.join(print_names) )
                 

               

                # remove unncessary terms if near optimzed solution is achieved or preprogrammed to do so
                err = [  (np.abs(outp[p.oname]-self.SARG[self.target_data[p.oname]] )) * self.SARG[self.target_mask[p.oname]]  for p in self.predColl.preds if p.pFunc is not None]
                errmax = np.max (  [ e.max() for e in err])
                try:
                    
                    if i>0 and ( (i//self.args.ITER2)%self.args.ITEM_REMOVE_ITER==0) :
                        print ( 'start removing non necessary clauses')
                        self.filter_predicates(session,'OR:0')  
                        self.filter_predicates(session,'AND:0')  
                except:
                    pass
                if  np.mean(cost)<self.args.FILT_TH_MEAN  and errmax<self.args.FILT_TH_MAX or ( np.mean(cost)<self.args.FILT_TH_MEAN and i%1000==0 ):
                    
                    should_remove=True
                    for ii in range(20):
                        cost,outp= self.runTSteps(session,False)
                        err = [  (np.abs(outp[p.oname]-self.SARG[self.target_data[p.oname]] )) * self.SARG[self.target_mask[p.oname]]  for p in self.predColl.outpreds if p.pFunc is not None]
                        errmax = np.max (  [ e.max() for e in err])

                
                        if  np.mean(cost)<self.args.FILT_TH_MEAN  and errmax <self.args.FILT_TH_MAX or ( np.mean(cost)<self.args.FILT_TH_MEAN and i%1000==0 ):
                            pass
                        else:
                            should_remove = False
                            break
                    should_remove = should_remove
                    if should_remove:
                        print ( 'start removing non necessary clauses')

                        self.filter_predicates(session,'OR:0')  
                        self.filter_predicates(session,'AND:0')  
                        if self.args.BINARAIZE==1:
                            self.binarize(session)
                            
                            self.filter_predicates(session,'OR')  
                            self.filter_predicates(session,'AND')  
                            cost,outp= self.runTSteps(session,False)
                        
                        if self.args.CHECK_CONVERGENCE==1:
                            self.check_weights(session,'AND:0') 
                            self.check_weights(session,'OR:0') 

                
                # display learned predicates
                if self.args.PRINTPRED :
                    try:
                        
                        for p in self.predColl.preds:

                            
                            if p.pFunc is None:
                                continue
                            
                            inp_list = p.inp_list  

                            if p.pFunc is not None:
                                s = p.pFunc.get_func( session,inp_list,threshold=self.args.W_DISP_TH)
                                if s is not None:
                                    if len(s)>0: 
                                        print( p.dname+ '(' + ','.join(variable_list[0:p.arity]) + ')  \n'+s)

                    except:
                        print('there was an exception in print pred')
                # display raw membership weights for predicates
                if self.args.PRINT_WEIGHTS==1:
                    
                    wts = tf.trainable_variables( )
                    wvs = session.run( wts )
                    for t,w in zip( wts,wvs):
                        if '_SM' in t.name:
                            print( t.name, np.squeeze( w.argmax(-1) ) )
                        else:
                            print( t.name, myC( p.pFunc.conv_weight_np(w) ,2) )
            
                
                
                # check for optimization
                err = [  (np.abs(outp[p.oname]-self.SARG[self.target_data[p.oname]] )) * self.SARG[self.target_mask[p.oname]]  for p in self.predColl.outpreds if p.pFunc is not None ]
                errmax = np.max (  [ e.max() for e in err])
                
                if np.mean(cost)<self.args.OPT_TH  and ( np.mean(cost)<.0 or  errmax<.09 ):
                    

                    if self.args.CHECK_CONVERGENCE==1:
                        if self.check_weights(session,'OR:0')  and self.check_weights(session,'AND:0')  :
                        
                            print('optimization finished !')
                            
                            return
                    else:
                        print('optimization finished !')
                            
                        return

                start_time=now


