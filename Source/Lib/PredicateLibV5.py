import  numpy as np
from itertools import product,permutations
from collections import OrderedDict 
from datetime import datetime
from collections import Counter

variable_list=['A','B','C','D','E','F','G','H','I','J','K','L','M','N']

#####################################################
def gen_all_orders2( v, r,var=False):
    if not var:
        inp = [v[i] for i in r]
    else:
        inp = [v[i[0]] for i in r]
        
    p = product( *inp)
    return [kk for kk in p]


###################################################
###################################################
###################################################

class Background:

    def __init__(self,predColl ):
        
        self.predColl = predColl
        self.backgrounds = OrderedDict({})
        self.backgrounds_value = OrderedDict({})
        self.examples = OrderedDict({})
        self.examples_value = OrderedDict({})
        self.backgrounds_ind = OrderedDict({})
        self.examples_ind = OrderedDict({})
        self.continuous_vals = OrderedDict({})

        for p in predColl.preds:
            self.backgrounds[p.oname] = []
            self.backgrounds_value[p.oname] = []
            self.backgrounds_ind[p.oname] = []
            self.examples[p.oname] = []
            self.examples_value[p.oname] = []
            self.examples_ind[p.oname] = []

    def add_backgroud( self,pred_name , pair ,value=1): 
        
        if pair not in self.backgrounds[pred_name]:
            self.backgrounds[pred_name].append( pair)
            self.backgrounds_value[pred_name].append(value)
            self.backgrounds_ind[pred_name].append( self.predColl.pairs[pred_name].index(pair))

    def add_example(self,pred_name,pair,value=1):
        if pair not in self.examples[pred_name]:
            self.examples[pred_name].append(pair)
            self.examples_value[pred_name].append(value)
            self.examples_ind[pred_name].append( self.predColl.pairs[pred_name].index(pair))

    def add_all_neg_example(self,pred_name):
        # pairs = gen_all_orders2(self.predColl.constants, self.predColl[pred_name].arguments )
        for pa in self.predColl.pairs[pred_name]:
            if  pa not in self.examples[pred_name] :#and pa not in self.backgrounds[pred_name] :
                self.add_example(pred_name,pa,0.)
    
    def add_all_neg_example_ratio(self,pred_name, ratio):
        # pairs = gen_all_orders2(self.predColl.constants, self.predColl[pred_name].arguments )
        pos = np.sum(self.examples_value[pred_name]) 
        max_neg = pos * ratio
        inds= np.random.permutation(self.predColl.pairs_len[pred_name] )
        
        cnt = 0 
        for i in inds:
            if  self.predColl.pairs[pred_name][i]  not in self.examples[pred_name] :#and pa not in self.backgrounds[pred_name] :
                cnt+=1
                self.add_example(pred_name,self.predColl.pairs[pred_name][i],0.)
            if cnt>max_neg:
                break

    def add_number_bg( self,N ,ops=['incN','zeroN', 'lteN','eqN', 'gtN'  ]):
        if 'zeroN' in ops :
            self.add_backgroud( 'zeroN' , ('0',) )
        
        for a in N:
            if 'incN' in ops:
                if  ( str(int(a)+1)) in N:
                    self.add_backgroud('incN', ( a, str(int(a)+1) ) )
            
        for a in N:
            for b in N:
                    if 'addN' in ops:
                        c = str( int(a)+int(b) )
                        if c in N:
                            self.add_backgroud('addN', (a,b,c))
                    
                    if 'eqN' in ops and a==b:
                        self.add_backgroud('eqN', (a , b))
                    if 'lteN' in ops and a<=b:
                        self.add_backgroud('lteN', (a , b))
                    if 'gtN' in ops and a>b:
                        self.add_backgroud('gtN',(a , b))

    def add_list_bg(self, C,Ls , ops=['emptyL','eqC', 'eqL','singleL'  ]):
        if 'LI' in ops:
            for a in Ls:
                for i in range(len(a)):
                    self.add_backgroud( 'LI', (a,a[i], str(i) ) )
        
        if 'LL' in ops:
            for a in Ls:
                self.add_backgroud( 'LL', (a, str(len(a)) ) )
              
        if 'emptyL' in ops:
            self.add_backgroud ('emptyL' , ('',) )
        
        for a in C:
            if 'eqC' in ops:
                self.add_backgroud ('eqC', (a,a) ) 
            if 'eqLC' in ops:
                if a in C and a in Ls:
                    self.add_backgroud ('eqLC', (a,a) ) 

        for a in Ls:
            if 'eqL' in ops:
                self.add_backgroud ('eqL', (a,a) ) 
            
            if len(a) ==1:
                if 'singleL' in ops:
                    self.add_backgroud ('singleL', (a,) ) 

            for b in Ls:
                if a+b in Ls:
                    if 'appendL' in ops:
                        self.add_backgroud ('appendL', (a,b,a+b) ) 
                    if 'appendC1' in ops and len(b)==1:
                        self.add_backgroud ('appendC1', (a,b,a+b) ) 
                    if 'appendC2' in ops and len(a)==1:
                        self.add_backgroud ('appendC2', (a,b,a+b) ) 

        return

    def add_continous_valuea( self,vdic):
        self.continuous_vals.update(vdic)
    
    def add_continous_value( self,key,value):
        self.continuous_vals[key]=value
    
    def get_X0(self,pred_name):
        try:
            x = np.zeros( [self.predColl.pairs_len[pred_name] ] , np.float32 ) 
            x[self.backgrounds_ind[pred_name]]=self.backgrounds_value[pred_name]
        except:
            print('exception', pred_name)
            exit(0)
        return x
    
    def get_target_data(self,pred_name):
        x = np.zeros( [self.predColl.pairs_len[pred_name] ] , np.float32 )
        x[self.examples_ind[pred_name]]=self.examples_value[pred_name]
        return x
        
    def get_target_mask(self,pred_name):
        x = np.zeros( [self.predColl.pairs_len[pred_name] ] , np.float32 )
        x[self.examples_ind[pred_name]]=1
        return x

###################################################
#####################################################
#####################################################

class Predicate:
    
    def __init__(self,dname,oname=None, arguments=[],variables=[],pFunc=None, inc_preds=None,exc_preds=None,use_cnt_vars=False, use_neg=False,arg_funcs=None,inc_cnt=None,exc_cnt=None,Fam='eq', exc_terms=[],exc_conds=[],max_T=100,chunk_count=0,pairs=None):
        
        self.dname=dname 
        self.oname=oname 
        if self.oname is None:
            self.oname=self.dname
  
            


        self.exc_term_inds={}
        self.arity=len(arguments)
        self.var_count = len(variables)
        self.arguments = arguments 
        self.variables = variables

        self.pFunc=pFunc
        self.use_neg = use_neg
        self.exc_cnt=exc_cnt
        self.inc_cnt=inc_cnt
        
        self.inc_preds=inc_preds
        self.exc_preds=exc_preds
        self.use_cnt_vars = use_cnt_vars

        self.exc_conds=exc_conds
        self.exc_terms=exc_terms
        
        self.arg_funcs = arg_funcs
        # if self.arg_funcs is not None:
        self.use_tH = (self.arg_funcs is not None and 'tH'in arg_funcs)
        self.use_Th = (self.arg_funcs is not None and 'Th'in arg_funcs)
        self.use_M = (self.arg_funcs is not None and 'M'in arg_funcs)
        self.use_P = (self.arg_funcs is not None and 'P'in arg_funcs)
        
        self.Fam = Fam
        

        self.inp_list=[]
        self.Lx=0
        self.Lx_details=[]
        self.Lx_details_dic={}
        
        self.max_T = max_T
        self.chunk_count=chunk_count
        self.pairs=pairs

    def get_term_index(self,term ):

        if not 'not ' in term:
            ind = self.inp_list.index(term)
        else:
            ind = self.inp_list.index(term[4:]) + self.Lx

        return ind

#####################################################
#####################################################
#####################################################

class PredFunc:
    def __init__(self,name="",trainable=True):
        self.trainable = trainable
        self.name=name
    
    def pred_func(self,xi,xcs=None,t=0):
        pass
    
    def get_func(self,session,names=None,threshold=.1,print_th=True):
        pass
    def get_item_contribution(self,session,names=None,threshold=.1):
        pass
    def conv_weight_np(self,w):
        return w
    def conv_weight(self,w):
        return w
 
#####################################################
#####################################################
#####################################################

class PredCollection:
    
    def __init__(self, constants  ):
        self.constants = constants
        self.outpreds=[]
        self.preds = []
        self.cnts=[]
        self.preds_by_name=dict({})

        
        self.pairs = OrderedDict()
        self.pairs_len  = OrderedDict()
        self.rev_pairs_index =  OrderedDict()


    def get_constant_list( self,pred , vl ):
        Cs=dict( { k:[] for k in self.constants.keys()})
        for i,cl in enumerate( pred.arguments+pred.variables ):
            Cs[cl[0]].append( vl[i])
            if cl[0]=='N':
                if pred.use_M: 
                    Cs['N'].append('M_' + vl[i] )
                if pred.use_P: 
                    Cs['N'].append('P_' + vl[i] )
                    
            if cl[0]=='L':
                
                if pred.use_tH:
                    Cs['L'].append('H_'+vl[i])
                    Cs['C'].append('t_'+vl[i])
                
                if pred.use_Th:
                    Cs['C'].append('h_'+vl[i])
                    Cs['L'].append('T_'+vl[i])
        
        return Cs
    def get_continous_var_names(self,p,thDictGT,thDictLT):
        terms=[]
        for v in self.cnts:
            
            cond1 = p.exc_cnt is not None and v.name in p.exc_cnt
            cond2 = p.inc_cnt is not None and v.name not in p.inc_cnt
            if not (cond1 or cond2):
                terms.extend( v.get_terms(thDictGT[v.name],thDictLT[v.name]) )
        return terms
    def get_continous_var_names_novar(self,p):
        terms=[]
        for v in self.cnts:
            
            cond1 = p.exc_cnt is not None and v.name in p.exc_cnt
            cond2 = p.inc_cnt is not None and v.name not in p.inc_cnt
            if not (cond1 or cond2):
                terms.extend( v.get_terms_novar() )
        return terms
    def add_counter( self):
        self.add_pred(dname='CNT',arguments=['N'] , variables=[] )   
    def add_continous( self, name,no_lt, no_gt,lt_init=None,gt_init=None,dim=1):
        self.cnts.append( ContinousVar(name,no_lt,no_gt,lt_init,gt_init,dim))
    def add_number_preds(self,ops=['incN','zeroN', 'lteN','eqN', 'gtN'   ]):
        
        if 'incN' in ops:
            self.add_pred(dname='incN'      ,arguments=['N','N'] , variables=[] )   

        if 'addN' in ops:
            self.add_pred(dname='addN'      ,arguments=['N','N','N'] , variables=[] )   

        if 'zeroN' in ops:
            self.add_pred(dname='zeroN'      ,arguments=['N'] , variables=[] )   
        if 'lteN' in ops:
            self.add_pred(dname='lteN'      ,arguments=['N','N'] , variables=[] )   

        if 'gtN' in ops:
            self.add_pred(dname='gtN'      ,arguments=['N','N'] , variables=[] )   

        if 'eqN' in ops:
            self.add_pred(dname='eqN'      ,arguments=['N','N'] , variables=[] )   
    def add_list_preds(self , ops=['emptyL','eqC', 'eqL','singleL'  ] ):
        
        if 'LI' in ops:
            self.add_pred(dname='LI'      ,arguments=['L', 'C','N'] , variables=[] )  # [_|t]-> ?
        if 'LL' in ops:
            self.add_pred(dname='LL'      ,arguments=['L', 'N'] , variables=[] )  # [_|t]-> ?
        
        if 'eqC' in ops:
            self.add_pred(dname='eqC'      ,arguments=['C','C'] , variables=[] )  # [_|t]-> ?
        if 'eqLC' in ops:
            self.add_pred(dname='eqLC'      ,arguments=['L','C'] , variables=[] )  # [_|t]-> ?
        if 'emptyL' in ops:
            self.add_pred(dname='emptyL'      ,arguments=['L'] , variables=[] )  # [_|t]-> ?
        if 'eqL' in ops:
            self.add_pred(dname='eqL'     ,arguments=['L','L'] , variables=[] )   
        if 'singleL' in ops:
            self.add_pred(dname='singleL'     ,arguments=['L'] , variables=[] )   
        if 'appendL' in ops:
            self.add_pred(dname='appendL'     ,arguments=['L','L','L'] , variables=[] )   
        if 'appendC1' in ops:
            self.add_pred(dname='appendC1'     ,arguments=['L','C','L'] , variables=[] )   
        if 'appendC2' in ops:
            self.add_pred(dname='appendC2'     ,arguments=['C','L','L'] , variables=[] )   
    def add_pred( self,**args):
        p = Predicate( **args)
        self.preds.append(p)
        not_in=True
        for pp in self.outpreds:
            if pp.oname == p.oname:
                not_in=False
        if not_in:
            self.outpreds.append(p)
        self.preds_by_name[p.dname] = p
        return p
    def __len__(self):
        return len(self.preds)
    def __getitem__(self, key):
        if type(key) in (str,):
            return self.preds_by_name[key]
        else:
            return self.preds[key]
    def apply_func_args(self,Cs):
        
        if 'C' in Cs:
            for i,v in enumerate(Cs['C']):
                if v.startswith('t_'):
                    if v=='t_':
                        Cs['C'][i] = ''
                    else:
                        Cs['C'][i]=v[-1]
        if 'L' in Cs:
            for i,v in enumerate(Cs['L']):
                if v.startswith('H_'):
                    if v=='H_':
                        Cs['L'][i] = ''
                    else:
                        Cs['L'][i]= v[2:-1]
        if 'C' in Cs:
            for i,v in enumerate(Cs['C']):
                if v.startswith('h_'):
                    if v=='h_':
                        Cs['C'][i] = ''
                    else:
                        Cs['C'][i]=v[2]
        if 'L' in Cs:
            for i,v in enumerate(Cs['L']):
                if v.startswith('T_'):
                    if v=='T_':
                        Cs['L'][i] = ''
                    else:
                        Cs['L'][i]= v[3:]
        if 'N' in Cs:
            for i,v in enumerate(Cs['N']):
                if type(v) in (str,) and v.startswith('P_'):
                    Cs['N'][i] = '%d'%( int(v[2:])+1)
                if type(v) in (str,) and v.startswith('M_'):
                    val= max(0,int(v[2:])-1)
                    Cs['N'][i] = '%d'%(val)
                    
        return Cs
    
    def map_fn( self,pair_arg,pair_val ,pred ):
             
             
            in_indices=[]
            L = 0
            Cs = self.get_constant_list(pred,pair_arg+pair_val)
            
            if pred.arg_funcs is not None:
                Cs = self.apply_func_args(Cs)

            for p in self.outpreds:
                
                
                # exclude some predciates
                if pred.Lx_details_dic[p.oname]==0:
                    continue
                if pred.inc_preds is not None and p.oname not in pred.inc_preds:
                    # L +=  self.pairs_len[p.oname] 
                    continue
                if pred.exc_preds is not None and p.oname in pred.exc_preds:
                    # L +=  self.pairs_len[p.oname] 
                    continue
                
                
                if len(p.arguments)>0:
                    name_set =  gen_all_orders2( Cs , p.arguments,var=True)
                else:
                    name_set=[()]
                
                
                for i,n in  enumerate(name_set):
                    if i in pred.exc_term_inds[p.oname]:
                        continue
                    try:
                        # ind = p.pairs.index(n)
                        ind = self.rev_pairs_index[p.oname][n]
                        in_indices.append(ind+L+1)
                    except:
                        in_indices.append(0)
                L +=  self.pairs_len[p.oname]
                
            return in_indices

    def initialize_predicates(self):
        
        t1=  datetime.now()

       
        # fill pred.pairs , pred.inp_list , pred.Lx , pred.Lx_details, pred.self_termIndex
        for pred in self.preds:
            
            
            if pred.oname not in self.pairs:
                # self.pairs[pred.oname]  = gen_all_orders2(self.constants,pred.arguments)
                # self.pairs_len[pred.oname] = len( self.pairs[pred.oname] )
                # pred.pairs.sort()  
            
                if pred.pairs is None:
                    self.pairs[pred.oname]  = gen_all_orders2(self.constants,pred.arguments)
                else:
                    self.pairs[pred.oname]  = pred.pairs

                self.pairs_len[pred.oname] = len( self.pairs[pred.oname] )


                self.rev_pairs_index[pred.oname]=OrderedDict()
                for ii in range(self.pairs_len[pred.oname] ):
                    self.rev_pairs_index[pred.oname][ self.pairs[pred.oname][ii]] = ii
            
            pred.Lx_details =[]
            Cs = self.get_constant_list(pred,variable_list)
            
            for p in self.outpreds:
                pred.exc_term_inds[p.oname]=[]    
                if pred.inc_preds is not None and p.oname not in pred.inc_preds:
                    pred.Lx_details.append(0)
                    pred.Lx_details_dic[p.oname]=0
                    continue
                if pred.exc_preds is not None and p.oname in pred.exc_preds:
                    pred.Lx_details.append(0)
                    pred.Lx_details_dic[p.oname]=0
                    continue

                if len(p.arguments)>0:
                    name_set =  gen_all_orders2( Cs , p.arguments,var=True)
                else:
                    name_set=[()]

                Li=0
                
                for i,n in enumerate(name_set):
                    term = p.oname + '(' + ','.join(n)+')'
                    pcond = False
                    for c in pred.exc_conds:
                        if p.oname == c[0] or c[0]=='*':
                            cl=Counter(n)
                            l = list(cl.values())
                            if len(l)>0:
                                if c[1]=='rep1':
                                    if max(l)>1:
                                        pcond=True
                                        break
                                if c[1]=='rep2':
                                    if max(l)>2:
                                        pcond=True
                                        break
                    if term not in pred.exc_terms and not pcond:
                        Li+=1
                        pred.inp_list.append( term)
                    else:
                        pred.exc_term_inds[p.oname].append(i)
                     
                pred.Lx_details.append(Li)
                pred.Lx_details_dic[p.oname]=Li
            
            if pred.use_neg:
                negs=[]
                for k in pred.inp_list:
                    negs.append( 'not '+k)  
                pred.inp_list.extend(negs)
            pred.Lx = sum(pred.Lx_details)
        
        

        
          
        self.values_sizes = OrderedDict({})
        for pred in self.preds:
            self.values_sizes[pred] = self.pairs_len[pred.oname]
        self.InputIndices=dict({})
         
        
        for p in self.preds:
            
            pairs_var = gen_all_orders2( self.constants , p.variables)
            len_pairs_var = len(pairs_var)

            self.InputIndices[p.dname] = np.zeros( [ self.pairs_len[p.oname], len_pairs_var , p.Lx], np.int64)
            
            if True:
                print('******************************************************************')
                print('predicate [%s,%s] parameters :'%(p.dname,p.oname) )
                print('Lx :',p.Lx)
                print('Lx Details',p.Lx_details )
                print('input index shape : ', self.InputIndices[p.dname].shape)
                print('******************************************************************')

            if p.pFunc is not None:    
                for i in range( self.pairs_len[p.oname] ):
                    for j in range( len_pairs_var ):
                        inds = self.map_fn( self.pairs[p.oname][i],pairs_var[j], p )
                        self.InputIndices[p.dname][i,j]=inds
                
                # self.InputIndices[p.dname]  = np.array(self.InputIndices[p.dname],np.int64)
            

        
        t2=  datetime.now()
        print ('building background knowledge finished. elapsed:' ,str(t2-t1))
    def get_terms_indexs(self,pred_name,terms):
        inds=[]
        terms=terms.split(', ')
        for t in terms:
            inds.append(self.preds_by_name[pred_name].inp_list.index(t))
        return np.array(inds,dtype=int)
            
        