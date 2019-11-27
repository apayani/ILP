from  Lib.ILPRLEngine import *
import argparse
from Lib.DNF import DNF

#define constants
E = [ 'a','b','c','d','e', 'f']
C = ['green','red']
Constants = dict( {'E':E ,'C':C})
predColl = PredCollection (Constants)

 

#define predicates
predColl.add_pred(dname='edge'  ,arguments=['E','E'])
predColl.add_pred(dname='color'  ,arguments=['E','C'])


predColl.add_pred(dname='target',arguments=['E'] , variables=['E','C'  ] , pFunc = DNF('target',terms=4,init=[-1,.1,-1,.1] ,sig=2)     )
predColl.initialize_predicates()    


all_bg=[]
#add first background
bg = Background( predColl )
bg.add_backgroud ('edge', ('a','b') ) 
bg.add_backgroud ('edge', ('b','c') )
bg.add_backgroud ('edge', ('b','d') )
bg.add_backgroud ('edge', ('c','e') )
bg.add_backgroud ('edge', ('e','f') )
bg.add_backgroud ('color', ('a','green') )
bg.add_backgroud ('color', ('b','red') )
bg.add_backgroud ('color', ('c','green') )
bg.add_backgroud ('color', ('d','green') )
bg.add_backgroud ('color', ('e','red') )
bg.add_backgroud ('color', ('f','red') )
  
bg.add_example('target',('e',))
bg.add_all_neg_example('target') 

all_bg.append(bg)


bg = Background( predColl )
bg.add_backgroud ('edge', ('a','b') ) 
bg.add_backgroud ('edge', ('b','a') )
bg.add_backgroud ('edge', ('c','b') )
bg.add_backgroud ('edge', ('f','c') )
bg.add_backgroud ('edge', ('d','c') )
bg.add_backgroud ('edge', ('e','d') )

bg.add_backgroud ('color', ('a','green') )
bg.add_backgroud ('color', ('b','green') )
bg.add_backgroud ('color', ('c','red') )
bg.add_backgroud ('color', ('d','green') )
bg.add_backgroud ('color', ('e','green') )
bg.add_backgroud ('color', ('f','red') )
  
bg.add_example('target',('a',))
bg.add_example('target',('b',))
bg.add_example('target',('e',))
bg.add_example('target',('f',))
bg.add_all_neg_example('target') 

all_bg.append(bg)






def bgs(it,is_training):
    return all_bg
 

###########################################################################


parser = argparse.ArgumentParser()

parser.add_argument('--CHECK_CONVERGENCE',default=1,help='Check for convergence',type=int)
parser.add_argument('--SHOW_PRED_DETAILS',default=1,help='Print predicates definition details',type=int)

parser.add_argument('--BS',default=2,help='Batch Size',type=int)
parser.add_argument('--T',default=5 ,help='Number of forward chain',type=int)
parser.add_argument('--LR_SC', default={ (-1000,2):.01 ,  (2,1e5):.01} , help='Learning rate schedule',type=dict)

parser.add_argument('--BINARAIZE', default=1 , help='Enable binrizing at fast convergence',type=int)
parser.add_argument('--MAX_DISP_ITEMS', default=10 , help='Max number  of facts to display',type=int)
parser.add_argument('--DISP_BATCH_VALUES',default=[],help='Batch Size',type=list)
parser.add_argument('--W_DISP_TH', default=.2 , help='Display Threshold for weights',type=int)
parser.add_argument('--ITER', default=400000, help='Maximum number of iteration',type=int)
parser.add_argument('--ITER2', default=200, help='Epoch',type=int)
parser.add_argument('--PRINTPRED',default=1,help='Print predicates',type=int)
parser.add_argument('--PRINT_WEIGHTS',default=0,help='Print raw weights',type=int)
parser.add_argument('--MAXTERMS',default=6 ,help='Maximum number of terms in each clause',type=int)
parser.add_argument('--L1',default=0 ,help='Penalty for maxterm',type=float)
parser.add_argument('--L2',default=0 ,help='Penalty for distance from binary for weights',type=float)
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
  
args = parser.parse_args()

print('displaying config setting...')
for arg in vars(args):
        print( '{}-{}'.format ( arg, getattr(args, arg) ) )
    

model = ILPRLEngine( args=args ,predColl=predColl ,bgs=bgs )
model.train_model()    


