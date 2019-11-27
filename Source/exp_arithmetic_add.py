from  Lib.ILPRLEngine import *
import argparse
import itertools
from Lib.DNF import DNF
from Lib.DNF_Ex import DNF_Ex


#define constants
maxN =5
C = []
for i in range(maxN+1):
    C.append( '%d'%i)
Constants = dict( {'P':C })

#define predicates
predColl = PredCollection (Constants)

predColl.add_pred(dname='eq'  ,arguments=['P','P'] )
predColl.add_pred(dname='zero',arguments=['P'] )
predColl.add_pred(dname='inc' ,arguments=['P','P'] )

predColl.add_pred(dname='add' ,arguments=['P','P','P'],variables=[ 'P','P']    , pFunc = DNF('add',terms=6,init=[-1,.1,-.1,.1],sig=2    ),arg_funcs=[ ] ,  Fam='or',exc_conds=[('*','rep1') ] , exc_terms=['add(A,B,C)'])
predColl.initialize_predicates()    


#add background
bg = Background( predColl)

for i in range(maxN+1):
    if i==0:
        bg.add_backgroud ('zero', ('%d'%i,)) 

    if i>0:
        bg.add_backgroud ('inc', ('%d'%(i-1),'%d'%(i)) ) 

for i in range(maxN+1):
    for j in range(maxN+1):
        if i==j:
            bg.add_backgroud ('eq', ('%d'%i,'%d'%j))

 
for i in range(maxN+1):
    for j in range(maxN+1):
        if i+j<=maxN:
            bg.add_example( 'add', ( '%d'%i,'%d'%j,'%d'%(i+j)))
    
bg.add_all_neg_example('add')
     

def bgs(it,is_training):
    return [bg,]
 
# bgs=[bg,]
###########################################################################


parser = argparse.ArgumentParser()
 
 

parser.add_argument('--CHECK_CONVERGENCE',default=1,help='Check for convergence',type=int)
parser.add_argument('--SHOW_PRED_DETAILS',default=1,help='Print predicates definition details',type=int)

parser.add_argument('--BS',default=1,help='Batch Size',type=int)
parser.add_argument('--T',default=6,help='Number of forward chain',type=int)
parser.add_argument('--LR_SC', default={ (-1000,2):.005 ,  (2,1e5):.003} , help='Learning rate schedule',type=dict)

parser.add_argument('--BINARAIZE', default=1 , help='Enable binrizing at fast convergence',type=int)
parser.add_argument('--MAX_DISP_ITEMS', default=10 , help='Max number  of facts to display',type=int)
parser.add_argument('--DISP_BATCH_VALUES',default=[],help='Batch Size',type=list)
parser.add_argument('--W_DISP_TH', default=.05 , help='Display Threshold for weights',type=int)
parser.add_argument('--ITER', default=400000, help='Maximum number of iteration',type=int)
parser.add_argument('--ITER2', default=200, help='Epoch',type=int)
parser.add_argument('--PRINTPRED',default=1,help='Print predicates',type=int)
parser.add_argument('--PRINT_WEIGHTS',default=0,help='Print raw weights',type=int)
parser.add_argument('--MAXTERMS',default=6 ,help='Maximum number of terms in each clause',type=int)
parser.add_argument('--L1',default=.001 ,help='Penalty for maxterm',type=float)
parser.add_argument('--L2',default=.001 ,help='Penalty for distance from binary',type=float)
parser.add_argument('--L3',default=0 ,help='Penalty for distance from binary for each term',type=float)
parser.add_argument('--L2LOSS',default=0,help='Use L2 instead of cross entropy',type=int)
parser.add_argument('--SYNC',default=0,help='Synchronized Update',type=int)
parser.add_argument('--ALLTIMESTAMP',default=0 ,help='Add loss for each timestamp',type=int)
parser.add_argument('--FILT_TH_MEAN', default=.5 , help='Fast convergence total loss threshold MEAN',type=float)
parser.add_argument('--FILT_TH_MAX', default=.5 , help='Fast convergence total loss threshold MAX',type=float)
parser.add_argument('--OPT_TH', default=.01 , help='Per value accuracy threshold',type=float)
parser.add_argument('--PLOGENT', default=.50 , help='Crossentropy coefficient',type=float)
parser.add_argument('--BETA1', default=.90 , help='ADAM Beta1',type=float)
parser.add_argument('--BETA2', default=.999 , help='ADAM Beta2',type=float)
parser.add_argument('--EPS', default=1e-6, help='ADAM Epsillon',type=float)
parser.add_argument('--GPU', default=1, help='Use GPU',type=int)
parser.add_argument('--LOGDIR', default='./logs/Logic', help='Log Dir',type=str)
parser.add_argument('--TB', default=0, help='Use Tensorboard',type=int)
parser.add_argument('--SEED',default=0,help='Random seed',type=int)
parser.add_argument('--ADDGRAPH', default=1, help='Add graph to Tensorboard',type=int)
parser.add_argument('--CLIP_NORM', default=0, help='Clip gradient',type=float)

args = parser.parse_args()

print('displaying config setting...')
for arg in vars(args):
        print( '{}-{}'.format ( arg, getattr(args, arg) ) )
    

model = ILPRLEngine( args=args ,predColl=predColl ,bgs=bgs )
model.train_model()    


