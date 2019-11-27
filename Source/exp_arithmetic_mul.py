from  Lib.ILPRLEngine import *
from Lib.DNF import DNF
from Lib.CNF import CNF
from Lib.DNF_Ex import DNF_Ex
import argparse

maxN=6

#define constants
N=[ '%d'%i for i in range(maxN+1)]
Constants = dict( {'N':N})
 

#define predicates
predColl = PredCollection (Constants)
ops=['addN','zeroN','incN']
predColl.add_number_preds( ops=ops)
predColl.add_pred(dname='mul' ,arguments=['N','N','N'],variables=[ 'N','N']    , pFunc = DNF('mul',terms=6,init=[-1,.1,-.1,.1],sig=2    ),arg_funcs=[ ] ,  Fam='or',exc_conds=[('*','rep1') ] , exc_terms=['mul(A,B,C)'])
predColl.initialize_predicates()    
 

#add background
bg = Background( predColl )
bg.add_number_bg(N,ops=ops )
for i in range(maxN +1):
    for j in range(maxN+1 ):
        for c in range(maxN +1):
            pa = ( '%d'%i,'%d'%j,'%d'%c)
            bg.add_example( 'mul', pa, float(c==i*j))


def bgs(it,is_training):
    return [bg,]
 
###########################################################################


parser = argparse.ArgumentParser()

parser.add_argument('--CHECK_CONVERGENCE',default=1,help='Print predicates definition details',type=int)
parser.add_argument('--SHOW_PRED_DETAILS',default=0,help='Print predicates definition details',type=int)
# parser.add_argument('--PRINTPRED',default=0,help='Print predicates',type=int)


parser.add_argument('--BS',default=1,help='Batch Size',type=int)
parser.add_argument('--T',default=7 ,help='Number of forward chain',type=int)
parser.add_argument('--LR_SC', default={ (-1000,2):.005 ,  (2,1e5):.004} , help='Learning rate schedule',type=dict)

parser.add_argument('--BINARAIZE', default=1 , help='Enable binrizing at fast convergence',type=int)
parser.add_argument('--MAX_DISP_ITEMS', default=10 , help='Max number  of facts to display',type=int)
parser.add_argument('--DISP_BATCH_VALUES',default=[],help='Batch Size',type=list)
parser.add_argument('--W_DISP_TH', default=.2 , help='Display Threshold for weights',type=int)
parser.add_argument('--ITER', default=400000, help='Maximum number of iteration',type=int)
parser.add_argument('--ITER2', default=200, help='Epoch',type=int)
parser.add_argument('--PRINTPRED',default=1,help='Print predicates',type=int)
parser.add_argument('--PRINT_WEIGHTS',default=0,help='Print raw weights',type=int)
parser.add_argument('--MAXTERMS',default=5 ,help='Maximum number of terms in each clause',type=int)
parser.add_argument('--L1',default=.001 ,help='Penalty for maxterm',type=float)
parser.add_argument('--L2',default=.001 ,help='Penalty for distance from binary for weights',type=float)
parser.add_argument('--L3',default=0 ,help='Penalty for distance from binary for each term',type=float)
parser.add_argument('--L2LOSS',default=0,help='Use L2 instead of cross entropy',type=int)
parser.add_argument('--SYNC',default=0,help='Synchronized Update',type=int)
parser.add_argument('--ALLTIMESTAMP',default=0 ,help='Add loss for each timestamp',type=int)
parser.add_argument('--FILT_TH_MEAN', default=.5 , help='Fast convergence total loss threshold MEAN',type=float)
parser.add_argument('--FILT_TH_MAX', default=.5 , help='Fast convergence total loss threshold MAX',type=float)
parser.add_argument('--OPT_TH', default=.05 , help='Per value accuracy threshold',type=float)
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
# parser.add_argument('--SYNC',default=0,help='Use L2 instead of cross entropy',type=int)
# parser.add_argument('--L2LOSS',default=0,help='Use L2 instead of cross entropy',type=int)
# parser.add_argument('--BS',default=1,help='Batch Size',type=int)
# parser.add_argument('--T',default=7 ,help='Number of forward chain',type=int)
# parser.add_argument('--LR_SC', default={ (-1000,2):.005 ,  (2,1e5):.001} , help='Learning rate schedule',type=dict)

# parser.add_argument('--MAXTERMS',default=5 ,help='Maximum number of terms in each clause',type=int)
# parser.add_argument('--L1',default=.001 ,help='Penalty for maxterm',type=float)
# parser.add_argument('--L2',default=.001,help='Penalty for distance from binary',type=float)
# parser.add_argument('--L3',default=0 ,help='Penalty for distance from binary for each term',type=float)
# parser.add_argument('--ALLTIMESTAMP',default=0 ,help='Add loss for each timestamp',type=int)
# parser.add_argument('--USE_OR',default=1 ,help='Use Or in updating value vectors',type=int)
 

# parser.add_argument('--FILT_TH_MEAN', default=4 , help='Fast convergence total loss threshold MEAN',type=float)
# parser.add_argument('--FILT_TH_MAX', default=.2 , help='Fast convergence total loss threshold MAX',type=float)
# parser.add_argument('--OPT_TH', default=.05 , help='Per value accuracy threshold',type=float)
# parser.add_argument('--PLOGENT', default=.50 , help='Crossentropy coefficient',type=float)
# parser.add_argument('--BETA1', default=.90 , help='ADAM Beta1',type=float)
# parser.add_argument('--BETA2', default=.999 , help='ADAM Beta2',type=float)
# parser.add_argument('--EPS', default=1e-6, help='ADAM Epsillon',type=float)
# parser.add_argument('--GPU', default=1, help='Use GPU',type=int)
# parser.add_argument('--ITER', default=400000, help='Maximum number of iteration',type=int)
# parser.add_argument('--ITER2', default=200, help='Epoch',type=int)
# parser.add_argument('--LOGDIR', default='./logs/Logic', help='Log Dir',type=str)
# parser.add_argument('--TB', default=0, help='Use Tensorboard',type=int)
# parser.add_argument('--ADDGRAPH', default=1, help='Add graph to Tensorboard',type=int)
# parser.add_argument('--CLIP_NORM', default=0, help='Clip gradient',type=float)
# parser.add_argument('--PRINTPRED',default=1,help='Print predicates',type=int)
# parser.add_argument('--PRINT_WEIGHTS',default=0,help='Print raw weights',type=int)
# parser.add_argument('--SEED',default=0,help='Random seed',type=int)
# parser.add_argument('--BINARAIZE', default=1 , help='Enable binrizing at fast convergence',type=int)
# parser.add_argument('--MAX_NEG_EX',default=0,help='Max negative examples',type=int)
# parser.add_argument('--MAX_POS_EX',default=0,help='Max positive examples',type=int)
# parser.add_argument('--MAX_DISP_ITEMS', default=200 , help='Max number  of facts to display',type=int)
# parser.add_argument('--DISP_BATCH_VALUES',default=[],help='Batch Size',type=list)
# parser.add_argument('--W_DISP_TH', default=.2 , help='Display Threshold for weights',type=int)

args = parser.parse_args()

print('displaying config setting...')
for arg in vars(args):
        print( '{}-{}'.format ( arg, getattr(args, arg) ) )
    

model = ILPRLEngine( args=args ,predColl=predColl ,bgs=bgs )
model.train_model()    

