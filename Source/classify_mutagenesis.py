from  Lib.ILPRLEngine import *
import argparse
from Lib.mylibw import read_by_tokens
from Lib.DNF import DNF
from Lib.CNF import CNF
from Lib.cgt import CGT

from Lib.PredicateLibV5 import PredFunc
from sklearn.metrics import accuracy_score ,precision_recall_curve,auc,precision_recall_fscore_support,average_precision_score,log_loss
from sklearn.metrics import  roc_auc_score ,precision_recall_curve,auc,precision_recall_fscore_support,accuracy_score,confusion_matrix
import pandas as pd
import csv
import operator
import scipy.signal
from sklearn.model_selection import KFold,GroupKFold

# for 5-fold we should run the program 5 times with TEST_SET_INDEX from 0 to 9



if True:
    parser = argparse.ArgumentParser()
    parser.add_argument('--TEST_SET_INDEX',default=9,help='0-9 the index of the 10-fold experiment',type=int)
    parser.add_argument('--CHECK_CONVERGENCE',default=0,help='Print predicates definition details',type=int)
    parser.add_argument('--SHOW_PRED_DETAILS',default=0,help='Print predicates definition details',type=int)
    parser.add_argument('--PRINTPRED',default=0,help='Print predicates',type=int)
    parser.add_argument('--SYNC',default=0, help='Use L2 instead of cross entropy',type=int)
    parser.add_argument('--L2LOSS',default=0,help='Use L2 instead of cross entropy',type=int)
    parser.add_argument('--BS',default=40,help='Batch Size',type=int)
    parser.add_argument('--T',default=5,help='Number of forward chain',type=int)
    parser.add_argument('--LR_SC', default={ (-1000,10):.05 , (8,10):.05,  (10,1e5):.05} , help='Learning rate schedule',type=dict)
    parser.add_argument('--ITEM_REMOVE_ITER',default=10000 ,help='length period of each item removal',type=int)
    parser.add_argument('--MAXTERMS',default=10 ,help='Maximum number of terms in each clause',type=int)
    
    
    parser.add_argument('--L1',default=.00 ,help='Penalty for maxterm',type=float)
    parser.add_argument('--L2',default=.00 ,help='Penalty for distance from binary',type=float)
    parser.add_argument('--L3',default=.00 ,help='Penalty for distance from binary',type=float)
    parser.add_argument('--ALLTIMESTAMP',default=0 ,help='Add loss for each timestamp',type=int)
    parser.add_argument('--LR', default=.003 , help='Base learning rate',type=float)
    parser.add_argument('--FILT_TH_MEAN', default=1 , help='Fast convergence total loss threshold MEAN',type=float)
    parser.add_argument('--FILT_TH_MAX', default=1 , help='Fast convergence total loss threshold MAX',type=float)
    parser.add_argument('--OPT_TH', default=1, help='Per value accuracy threshold',type=float)
    parser.add_argument('--PLOGENT', default=.5 , help='Crossentropy coefficient',type=float)
    parser.add_argument('--BETA1', default=.90 , help='ADAM Beta1',type=float)
    parser.add_argument('--BETA2', default=.999 , help='ADAM Beta2',type=float)
    parser.add_argument('--EPS', default=1e-6, help='ADAM Epsillon',type=float)
    parser.add_argument('--GPU', default=1, help='Use GPU',type=int)
    parser.add_argument('--ITER', default=150, help='Maximum number of iteration',type=int)
    parser.add_argument('--ITER2', default=10, help='Epoch',type=int)
    parser.add_argument('--LOGDIR', default='./logs/Logic', help='Log Dir',type=str)
    parser.add_argument('--TB', default=0, help='Use Tensorboard',type=int)
    parser.add_argument('--ADDGRAPH', default=1, help='Add graph to Tensorboard',type=int)
    parser.add_argument('--CLIP_NORM', default=0, help='Clip gradient',type=float)

    parser.add_argument('--PRINT_WEIGHTS',default=0,help='Print raw weights',type=int)
    parser.add_argument('--SEED',default=1,help='Random seed',type=int)
    parser.add_argument('--BINARAIZE', default=0 , help='Enable binrizing at fast convergence',type=int)
    parser.add_argument('--MAX_DISP_ITEMS', default=50 , help='Max number  of facts to display',type=int)
    parser.add_argument('--W_DISP_TH', default=.1 , help='Display Threshold for weights',type=int)
    parser.add_argument('--DISP_BATCH_VALUES',default=[],help='Batch Size',type=list)
    args = parser.parse_args()

    print('displaying config setting...')
    for arg in vars(args):
            print( '{}-{}'.format ( arg, getattr(args, arg) ) )
    


   
  


################################################

Classes={}



atoms=pd.read_csv( './data/muta/mine/atom.csv')
bonds=pd.read_csv( './data/muta/mine/bond.csv')
molecules=pd.read_csv( './data/muta/mine/molecule.csv')




#extract the constants from the facts and exaples in the datafiles
################################################



#defining the predicate functions
################################################

MAX_ATOM = 40

# MAX_TYPES = 36
LEVELS_LOGP=4
LEVELS_LUMO=4
LEVELS_CHARGE=20
LEVELS_CHARGE_DIFF=0

ATOMS=[ '%d'%(i+1) for i in range(MAX_ATOM)]

ELEMENTS = list( atoms['element'].unique() )
ATOM_TYPES = list( atoms['type'].unique() )
BOND_TYPES = list( bonds['type'].unique() )


# Molecules=[ '%d'%(i+1) for i in range(MAX_Molecules)]


Constants = { 'A':ATOMS, 'T':ATOM_TYPES, 'B':BOND_TYPES, 'E':ELEMENTS}
predColl = PredCollection (Constants)
 

predColl.add_pred(dname='ind1',arguments=[]  )
predColl.add_pred(dname='inda',arguments=[]  )

predColl.add_pred(dname='ae',arguments=['A','E'] )
predColl.add_pred(dname='at',arguments=['A','T'] )
predColl.add_pred(dname='ab',arguments=['A','A','B'] )
# predColl.add_pred(name='abany',arguments=['A','A'] )


for e in ELEMENTS:
    predColl.add_pred( dname='element_%s'%(str(e)) ,arguments=['E'],  )
 

for b in BOND_TYPES:
    predColl.add_pred( dname='bondtype_%s'%(str(b)) ,arguments=['B'],  )

# for t in ATOM_TYPES:
#     predColl.add_pred( name='atomtype_%s'%(str(t)) ,arguments=['T'],  )

inits=np.linspace(-1,1,LEVELS_LOGP).astype(np.float32)
for n in range(LEVELS_LOGP):
    predColl.add_pred(dname='logp_%d'%n,arguments=[] ,pFunc = CGT(name='logp_%d'%n, init_w=inits[n],c=20.0) , Fam='eq', max_T=1,inc_preds=['logp_%d'%n])

inits=np.linspace(-1,1,LEVELS_LUMO).astype(np.float32)
for n in range(LEVELS_LUMO):
    predColl.add_pred(dname='lumo_%d'%n,arguments=[] ,pFunc = CGT(name='lumo_%d'%n, init_w=inits[n],c=20.0) , Fam='eq', max_T=1,inc_preds=['lumo_%d'%n])

inits=np.linspace(-1,1,LEVELS_CHARGE).astype(np.float32)
for n in range(LEVELS_CHARGE):
    predColl.add_pred(dname='charge_%d'%n,arguments=['A'] ,pFunc = CGT(name='charge_%d'%n, init_w=inits[n],c=20.0) , Fam='eq', max_T=1,inc_preds=['charge_%d'%n])


# for n in range(LEVELS_CHARGE_DIFF):
#     predColl.add_pred(name='charge_diff_%d'%n,arguments=['A','A'] ,pFunc = CGT(name='charge_diff_%d'%n, init_w=np.random.random_sample(),c=4.0) , Fam='eq', max_T=1,inc_preds=['charge_diff_%d'%n], exc_terms=['charge_diff_%d(A,A)'%n,'charge_diff_%d(B,A)'%n,'charge_diff_%d(B,B)'%n])



# predColl.add_pred(name='charge_gt',arguments=['A','A'] )
 
incs=[]
aux_cnt=0
predColl.add_pred(dname='aux_%d'%aux_cnt,arguments=['A'], variables=['A','B'] ,pFunc = 
    DNF('aux_%d'%aux_cnt,terms=2,init=[-1,.1,-1,.1],sig=1)  , use_neg=False, exc_conds=[ ] , exc_terms=[],  Fam='or',chunk_count=0) 

aux_cnt+=1
predColl.add_pred(dname='aux_%d'%aux_cnt,arguments=['A'], variables=['A','E'] ,pFunc = 
    DNF('aux_%d'%aux_cnt,terms=2,init=[-1,.1,-1,.1],sig=1)  , use_neg=False, exc_conds=[ ] , exc_terms=[],  Fam='or',chunk_count=0) 
 

# aux_cnt+=1
# predColl.add_pred(name='aux_%d'%aux_cnt,arguments=['A'], variables=['T'] ,pFunc = 
#     DNF('aux_%d'%aux_cnt,terms=2,init=[-1,.1,-1,.1],sig=1)  , use_neg=False, exc_conds=[ ] , exc_terms=[],  Fam='or',chunk_count=5) 
 
# incs.append('active')
predColl.add_pred(dname='active',arguments=[], variables=['A','E','B'] ,pFunc = 
    DNF('active',terms=3,init=[-1,.1,-1,.1],sig=1 ,predColl=predColl)  , use_neg=True, exc_conds=[] , exc_terms=[],  Fam='or',chunk_count=0   ) 
    

predColl.initialize_predicates()    



 

# defining 5 background knowledge structures corresponding to 5 experiments

bgs_pos=[] 
bgs_neg=[] 

for i,row_molecule in molecules.iterrows():
    bg = Background( predColl ) 

    for e in ELEMENTS:
        bg.add_backgroud( 'element_%s'%(str(e)) , (e,)  )

    for b in BOND_TYPES:
        bg.add_backgroud( 'bondtype_%s'%(str(b)) , (b,)  )

    # for t in ATOM_TYPES:
    #     bg.add_backgroud( 'atomtype_%s'%(str(t)) , (t,)  )

    molecule_id = row_molecule['molecule_id']

    bg.add_backgroud( 'inda', (), value = float(row_molecule['inda']))
    bg.add_backgroud( 'ind1', (), value = float(row_molecule['ind1']))
    
    for n in range(LEVELS_LOGP):
        bg.add_backgroud( 'logp_%d'%n, (), value = float(row_molecule['logp']))
    for n in range(LEVELS_LUMO):
        bg.add_backgroud( 'lumo_%d'%n, (), value = float(row_molecule['lumo']))
    
    df_mask = atoms['molecule_id']==molecule_id
    df = atoms[df_mask]

    
    for j, row_atom in df.iterrows():
        atom_id =row_atom['atom_id'].split('_')[1]


        # bg.add_backgroud( 'element_any_%s'%(str(row_atom['element'])) , ()  )

        bg.add_backgroud('ae', (atom_id,row_atom['element']) )
        bg.add_backgroud('at', (atom_id,row_atom['type']) )

        for n in range(LEVELS_CHARGE):
            bg.add_backgroud( 'charge_%d'%n, (atom_id,), value = float(row_atom['charge']))

        df_mask = bonds['atom1_id'] == row_atom['atom_id']
        df_bond = bonds[df_mask]
        
        
        for k,row_bond in df_bond.iterrows():
            atom_id2 = row_bond['atom2_id'].split('_')[1]
            bg.add_backgroud('ab', (atom_id,atom_id2, row_bond['type']) )
            # bg.add_backgroud('ab', (atom_id2,atom_id, row_bond['type']) )
            # bg.add_backgroud('abany', (atom_id,atom_id2) )
            # bg.add_backgroud('abany', (atom_id2,atom_id) )

        for jj, row_atomj in df.iterrows():
            atom_id2 =row_atomj['atom_id'].split('_')[1]
            # bg.add_backgroud ( 'charge_gt', (atom_id,atom_id2), float(row_atom['charge']>row_atomj['charge']))
            
            for n in range(LEVELS_CHARGE_DIFF):
                bg.add_backgroud( 'charge_diff_%d'%n, (atom_id,atom_id2), value = float(row_atom['charge']-row_atomj['charge']))
                bg.add_backgroud( 'charge_diff_%d'%n, (atom_id2,atom_id), value = float(row_atomj['charge']-row_atom['charge']))



    bg.add_example('active', (), float(row_molecule['mutagenic']=='yes'))

    if row_molecule['mutagenic']=='yes':
        bgs_pos.append(bg)
    else:
        bgs_neg.append(bg)



inds_pos=np.arange(len(bgs_pos))
inds_neg=np.arange(len(bgs_neg))

kf = KFold(10,True,args.SEED)
pos_gr = kf.split(inds_pos)
neg_gr = kf.split(inds_neg)

Folds_pos=[]
Folds_neg=[]
for tr, te in pos_gr:
    Folds_pos.append( (tr,te))
for tr, te in neg_gr:
    Folds_neg.append( (tr,te))

 
train_data = [ bgs_pos[i] for i in  Folds_pos[args.TEST_SET_INDEX][0] ] +   [ bgs_neg[i] for  i in Folds_neg[args.TEST_SET_INDEX][0] ]
test_data = [ bgs_pos[i] for i in  Folds_pos[args.TEST_SET_INDEX][1] ] +   [ bgs_neg[i] for  i in Folds_neg[args.TEST_SET_INDEX][1] ]
train_data_pos = [ bgs_pos[i] for i in  Folds_pos[args.TEST_SET_INDEX][0] ] 
train_data_neg = [ bgs_neg[i] for  i in Folds_neg[args.TEST_SET_INDEX][0] ]

print(len(train_data))
print(len(test_data))
 
 
    
     
    
     
inds=np.arange(len(train_data)) 
ps=np.ones_like(inds)

# this is the callback function that provides the training algorithm with different background knowledge for training and testing
###############################################
last_inds = None
def bgs(it,is_train):
     
    if is_train:
        inds_pos=np.random.permutation(len(train_data_pos))
        inds_neg=np.random.permutation(len(train_data_neg))
        
        return [ train_data_pos[inds_pos[i]]  for i in range(10)] + [ train_data_neg[inds_neg[i]]  for i in range(10)]
  
    else:
        return test_data

    
    
# this callback function is called for custom display each time a testing background is gonna be tested
# ###########################################################################


def disp_fn(eng,it,session,cost,outp):
    


    o = outp['active']
    
    true_class = []
    pred_class = []
    
    for i in range(len(o)):
    
        true_class.append(    float( test_data[i].get_target_data('active') ) )
        pred_class.append(  float( o[i][0]) )
    

    true_class = np.array(true_class)
    pred_class = np.array(pred_class)
    
    

    
    
    auroc = roc_auc_score(true_class, pred_class)
    x,y,th1 = precision_recall_curve(true_class, pred_class)
    aupr = auc(y,x)
 
    acc = accuracy_score(true_class, (pred_class>=.5).astype(np.float) )
    
    print('--------------------------------------------------------')    
    print('acc : %.4f , AUROC : %.4f,  AUPR : %.4f'% (acc,auroc,aupr ) )

 
    return

     

model = ILPRLEngine( args=args ,predColl=predColl ,bgs=bgs ,disp_fn=disp_fn)
model.train_model()    


