from  Lib.ILPRLEngine import *
from Lib.DNF import DNF
import argparse

import itertools
C=['a','b','c' ,'d' ]
L1 = C #+ ['',]
L2 = list( ''.join(a) for a in itertools.permutations('abcd', 2) )
L3 = list( ''.join(a) for a in itertools.permutations('abcd', 3) )
Ls = list( set(L1+L2+L3  ))
Constants = dict( {'C':C,'L':(Ls  ) ,'L2':L2 })
list_ops=ops = ['eqC','eqL'  ]



predColl = PredCollection (Constants)
predColl.add_list_preds( ops=list_ops)

predColl.add_pred(dname='gt' ,arguments=['C','C'] , variables=[] )
predColl.add_pred(dname='lte' ,arguments=['C','C'] , variables=[] )
p=predColl.add_pred(dname='sort' ,arguments=['L','L'] , variables=['L2','L2']    , pFunc = DNF('sort',terms=4,init=[-1,.2,-2,.1],sig=2),arg_funcs='tH' )

pairs_bg=[]
pairs_ex=dict()

for a in Ls:
    sa=''.join(sorted(a)) 
    if  sa in Ls and len(a)>2:
        pairs_ex[(a,sa)]=1
        
        ct=0
        for b in L3[:] :
            if b!=sa and ct<10:
                ind = np.random.randint(len(L3))
                v = b
                if( v!=sa ):
                    pairs_ex[ (a,v)]=0
                    
                    ct+=1
        
    else:
        if sa in Ls:
            pairs_bg.append( (a,sa))

p.pairs=[]
for item in pairs_bg:
    p.pairs.append(item)
for item in pairs_ex:
    p.pairs.append(item)
p.pairs = list( set ( p.pairs))
p.pairs_len=len(p.pairs)

predColl.initialize_predicates()    
 
bg = Background( predColl )
bg.add_list_bg(C,Ls ,ops=list_ops)

bg.add_backgroud( 'lte', ('a','a'))
bg.add_backgroud( 'lte', ('a','b'))
bg.add_backgroud( 'lte', ('a','c'))

bg.add_backgroud( 'lte', ('b','b'))
bg.add_backgroud( 'lte', ('b','c'))

bg.add_backgroud( 'lte', ('c','c'))

bg.add_backgroud( 'gt', ('b','a'))
bg.add_backgroud( 'gt', ('c','a'))
bg.add_backgroud( 'gt', ('c','b'))


if 'd' in C:
    bg.add_backgroud( 'lte', ('a','d'))
    bg.add_backgroud( 'lte', ('b','d'))
    bg.add_backgroud( 'lte', ('c','d'))
    bg.add_backgroud( 'lte', ('d','d'))
    bg.add_backgroud( 'gt', ('d','a'))
    bg.add_backgroud( 'gt', ('d','b'))
    bg.add_backgroud( 'gt', ('d','c'))
 

for item in pairs_ex:
    bg.add_example('sort', item, pairs_ex[item] )
for item in pairs_bg:
    bg.add_backgroud('sort', item)


def bgs(it,is_train):
    return [bg,]
 
###########################################################################


parser = argparse.ArgumentParser()

parser.add_argument('--CHECK_CONVERGENCE',default=1,help='Check for convergence',type=int)
parser.add_argument('--SHOW_PRED_DETAILS',default=1,help='Print predicates definition details',type=int)

parser.add_argument('--SYNC',default=0,help='Use L2 instead of cross entropy',type=int)
parser.add_argument('--L2LOSS',default=0,help='Use L2 instead of cross entropy',type=int)
parser.add_argument('--BS',default=1,help='Batch Size',type=int)
parser.add_argument('--T',default=1 ,help='Number of forward chain',type=int)
parser.add_argument('--LR_SC', default={ (-1000,2):.005 ,  (2,1e5):.002} , help='Learning rate schedule',type=dict)

parser.add_argument('--MAXTERMS',default=8 ,help='Maximum number of terms in each clause',type=int)
parser.add_argument('--L1',default=.001 ,help='Penalty for maxterm',type=float)
parser.add_argument('--L2',default=.001 ,help='Penalty for distance from binary',type=float)
parser.add_argument('--L3',default=0 ,help='Penalty for distance from binary for each term',type=float)
parser.add_argument('--ALLTIMESTAMP',default=0 ,help='Add loss for each timestamp',type=int)
parser.add_argument('--USE_OR',default=1 ,help='Use Or in updating value vectors',type=int)
parser.add_argument('--SIG',default=2,help='sigmoid coefficient',type=int)

parser.add_argument('--N1',default=1,help='softmax N1',type=int)
parser.add_argument('--N2',default=1,help='Softmax N2',type=int)
parser.add_argument('--FILT_TH_MEAN', default=.5 , help='Fast convergence total loss threshold MEAN',type=float)
parser.add_argument('--FILT_TH_MAX', default=.5 , help='Fast convergence total loss threshold MAX',type=float)
parser.add_argument('--OPT_TH', default=.05 , help='Per value accuracy threshold',type=float)
parser.add_argument('--PLOGENT', default=.50 , help='Crossentropy coefficient',type=float)
parser.add_argument('--BETA1', default=.90 , help='ADAM Beta1',type=float)
parser.add_argument('--BETA2', default=.999 , help='ADAM Beta2',type=float)
parser.add_argument('--EPS', default=1e-6, help='ADAM Epsillon',type=float)
parser.add_argument('--GPU', default=1, help='Use GPU',type=int)
parser.add_argument('--ITER', default=400000, help='Maximum number of iteration',type=int)
parser.add_argument('--ITER2', default=200, help='Epoch',type=int)
parser.add_argument('--LOGDIR', default='./logs/Logic', help='Log Dir',type=str)
parser.add_argument('--TB', default=0, help='Use Tensorboard',type=int)
parser.add_argument('--ADDGRAPH', default=1, help='Add graph to Tensorboard',type=int)
parser.add_argument('--CLIP_NORM', default=0, help='Clip gradient',type=float)
parser.add_argument('--PRINTPRED',default=1,help='Print predicates',type=int)
parser.add_argument('--PRINT_WEIGHTS',default=0,help='Print raw weights',type=int)
parser.add_argument('--SEED',default=0,help='Random seed',type=int)
parser.add_argument('--BINARAIZE', default=1 , help='Enable binrizing at fast convergence',type=int)
parser.add_argument('--MAX_NEG_EX',default=0,help='Max negative examples',type=int)
parser.add_argument('--MAX_POS_EX',default=0,help='Max positive examples',type=int)
parser.add_argument('--MAX_DISP_ITEMS', default=10 , help='Max number  of facts to display',type=int)
parser.add_argument('--DISP_BATCH_VALUES',default=[],help='Batch Size',type=list)
parser.add_argument('--W_DISP_TH', default=.2 , help='Display Threshold for weights',type=int)

args = parser.parse_args()

print('displaying config setting...')
for arg in vars(args):
        print( '{}-{}'.format ( arg, getattr(args, arg) ) )
    

model = ILPRLEngine( args=args ,predColl=predColl ,bgs=bgs )
model.train_model()    


