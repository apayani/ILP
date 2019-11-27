from  Lib.ILPRLEngine import *
from Lib.DNF import DNF
import argparse


#define constants
C=['Christopher','Arthur','Victoria' ,'Andrew', 'James' ,'Colin', 'Jennifer' , 'Charlotte', 'Emilio', 'Roberto', 'Lucia' , 'Pierro', 'Marco' , 'Angela' , 'Alfonso' , 'Sophia','Penelope','Christine','Maria','Francesca','Margaret','Charles','Gina','Tomaso']
C=list(set(C))
Constants = dict( {'C':C })
 
#define predicates
predColl = PredCollection (Constants)
predColl.add_pred(dname='wife'   ,arguments=['C','C'] )
predColl.add_pred(dname='husband'   ,arguments=['C','C'] )
predColl.add_pred(dname='father'   ,arguments=['C','C'] )
predColl.add_pred(dname='daughter'   ,arguments=['C','C'] )
predColl.add_pred(dname='mother'   ,arguments=['C','C'] )
predColl.add_pred(dname='son'   ,arguments=['C','C'] )
predColl.add_pred(dname='sister'   ,arguments=['C','C'] )
predColl.add_pred(dname='brother'   ,arguments=['C','C'] )
predColl.add_pred(dname='aunt'   ,arguments=['C','C'] )
predColl.add_pred(dname='uncle'   ,arguments=['C','C'] )
predColl.add_pred(dname='niece'   ,arguments=['C','C'] )
predColl.add_pred(dname='grandparent' ,arguments=['C','C'] )
predColl.add_pred(dname='nephew'   ,arguments=['C','C'] , variables=['C'] ,  pFunc = DNF('nephew',terms=4,init=[1,-1,-2,.1],sig=2) )
 
predColl.initialize_predicates()    
 

#add background
bg = Background( predColl )

bg.add_backgroud ('father', ('Christopher','Arthur'))
bg.add_backgroud ('father', ('Christopher', 'Victoria'))
bg.add_backgroud ('father', ('Andrew', 'James'))
bg.add_backgroud ('father', ('Andrew', 'Jennifer'))
bg.add_backgroud ('father', ('James', 'Colin'))
bg.add_backgroud ('father', ('James', 'Charlotte'))
bg.add_backgroud ('father', ('Roberto', 'Emilio'))
bg.add_backgroud ('father', ('Roberto', 'Lucia'))
bg.add_backgroud ('father', ('Pierro', 'Marco'))
bg.add_backgroud ('father', ('Pierro', 'Angela'))
bg.add_backgroud ('father', ('Marco', 'Alfonso'))
bg.add_backgroud ('father', ('Marco', 'Sophia'))

bg.add_backgroud ('mother', ('Penelope', 'Arthur'))
bg.add_backgroud ('mother', ('Penelope', 'Victoria'))
bg.add_backgroud ('mother', ('Christine', 'James'))
bg.add_backgroud ('mother', ('Christine', 'Jennifer'))
bg.add_backgroud ('mother', ('Victoria', 'Colin'))
bg.add_backgroud ('mother', ('Victoria', 'Charlotte'))
bg.add_backgroud ('mother', ('Maria', 'Emilio'))
bg.add_backgroud ('mother', ('Maria', 'Lucia'))
bg.add_backgroud ('mother', ('Francesca', 'Marco'))
bg.add_backgroud ('mother', ('Francesca', 'Angela'))
bg.add_backgroud ('mother', ('Lucia', 'Alfonso'))
bg.add_backgroud ('mother', ('Lucia', 'Sophia'))

bg.add_backgroud ('husband', ('Christopher', 'Penelope'))
bg.add_backgroud ('husband', ('Andrew', 'Christine'))
bg.add_backgroud ('husband', ('Arthur', 'Margaret'))
bg.add_backgroud ('husband', ('James', 'Victoria'))
bg.add_backgroud ('husband', ('Charles', 'Jennifer'))
bg.add_backgroud ('husband', ('Roberto', 'Maria'))
bg.add_backgroud ('husband', ('Pierro', 'Francesca'))
bg.add_backgroud ('husband', ('Emilio', 'Gina'))
bg.add_backgroud ('husband', ('Marco', 'Lucia'))
bg.add_backgroud ('husband', ('Tomaso', 'Angela'))

bg.add_backgroud ('wife', ('Penelope', 'Christopher'))
bg.add_backgroud ('wife', ('Christine', 'Andrew'))
bg.add_backgroud ('wife', ('Margaret', 'Arthur'))
bg.add_backgroud ('wife', ('Victoria', 'James'))
bg.add_backgroud ('wife', ('Jennifer', 'Charles'))
bg.add_backgroud ('wife', ('Maria', 'Roberto'))
bg.add_backgroud ('wife', ('Francesca', 'Pierro'))
bg.add_backgroud ('wife', ('Gina', 'Emilio'))
bg.add_backgroud ('wife', ('Lucia', 'Marco'))
bg.add_backgroud ('wife', ('Angela', 'Tomaso'))

bg.add_backgroud ('son', ('Arthur', 'Christopher'))
bg.add_backgroud ('son', ('Arthur', 'Penelope'))
bg.add_backgroud ('son', ('James', 'Andrew'))
bg.add_backgroud ('son', ('James', 'Christine'))
bg.add_backgroud ('son', ('Colin', 'Victoria'))
bg.add_backgroud ('son', ('Colin', 'James'))
bg.add_backgroud ('son', ('Emilio', 'Roberto'))
bg.add_backgroud ('son', ('Emilio', 'Maria'))
bg.add_backgroud ('son', ('Marco', 'Pierro'))
bg.add_backgroud ('son', ('Marco', 'Francesca'))
bg.add_backgroud ('son', ('Alfonso', 'Lucia'))
bg.add_backgroud ('son', ('Alfonso', 'Marco'))

bg.add_backgroud ('daughter', ('Victoria', 'Christopher'))
bg.add_backgroud ('daughter', ('Victoria', 'Penelope'))
bg.add_backgroud ('daughter', ('Jennifer', 'Andrew'))
bg.add_backgroud ('daughter', ('Jennifer', 'Christine'))
bg.add_backgroud ('daughter', ('Charlotte', 'Victoria'))
bg.add_backgroud ('daughter', ('Charlotte', 'James'))
bg.add_backgroud ('daughter', ('Lucia', 'Roberto'))
bg.add_backgroud ('daughter', ('Lucia', 'Maria'))
bg.add_backgroud ('daughter', ('Angela', 'Pierro'))
bg.add_backgroud ('daughter', ('Angela', 'Francesca'))
bg.add_backgroud ('daughter', ('Sophia', 'Lucia'))
bg.add_backgroud ('daughter', ('Sophia', 'Marco'))

bg.add_backgroud ('brother', ('Arthur', 'Victoria'))
bg.add_backgroud ('brother', ('James', 'Jennifer'))
bg.add_backgroud ('brother', ('Colin', 'Charlotte'))
bg.add_backgroud ('brother', ('Emilio', 'Lucia'))
bg.add_backgroud ('brother', ('Marco', 'Angela'))
bg.add_backgroud ('brother', ('Alfonso', 'Sophia'))

bg.add_backgroud ('sister', ('Victoria', 'Arthur'))
bg.add_backgroud ('sister', ('Jennifer', 'James'))
bg.add_backgroud ('sister', ('Charlotte', 'Colin'))
bg.add_backgroud ('sister', ('Lucia', 'Emilio'))
bg.add_backgroud ('sister', ('Angela', 'Marco'))
bg.add_backgroud ('sister', ('Sophia', 'Alfonso'))

bg.add_backgroud ('uncle', ('Arthur', 'Colin'))
bg.add_backgroud ('uncle', ('Charles', 'Colin'))
bg.add_backgroud ('uncle', ('Arthur', 'Charlotte'))
bg.add_backgroud ('uncle', ('Charles', 'Charlotte'))
bg.add_backgroud ('uncle', ('Emilio', 'Alfonso'))
bg.add_backgroud ('uncle', ('Tomaso', 'Alfonso'))
bg.add_backgroud ('uncle', ('Emilio', 'Sophia'))
bg.add_backgroud ('uncle', ('Tomaso', 'Sophia'))

bg.add_backgroud ('aunt', ('Jennifer', 'Colin'))
bg.add_backgroud ('aunt', ('Margaret', 'Colin'))
bg.add_backgroud ('aunt', ('Jennifer', 'Charlotte'))
bg.add_backgroud ('aunt', ('Margaret', 'Charlotte'))
bg.add_backgroud ('aunt', ('Angela', 'Alfonso'))
bg.add_backgroud ('aunt', ('Gina', 'Alfonso'))
bg.add_backgroud ('aunt', ('Angela', 'Sophia'))
bg.add_backgroud ('aunt', ('Gina', 'Sophia'))

bg.add_example ('nephew', ('Colin', 'Arthur'))
bg.add_example ('nephew', ('Colin', 'Jennifer'))
bg.add_example ('nephew', ('Alfonso', 'Emilio'))
bg.add_example ('nephew', ('Alfonso', 'Angela'))
bg.add_example ('nephew', ('Colin', 'Margaret'))
bg.add_example ('nephew', ('Colin', 'Charles'))
bg.add_example ('nephew', ('Alfonso', 'Gina'))
bg.add_example ('nephew', ('Alfonso', 'Tomaso'))


 
bg.add_backgroud ('niece', ('Charlotte', 'Arthur'))
bg.add_backgroud ('niece', ('Charlotte', 'Jennifer'))
bg.add_backgroud ('niece', ('Sophia', 'Emilio'))
bg.add_backgroud ('niece', ('Sophia', 'Angela'))
bg.add_backgroud ('niece', ('Charlotte', 'Margaret'))
bg.add_backgroud ('niece', ('Charlotte', 'Charles'))
bg.add_backgroud ('niece', ('Sophia', 'Gina'))
bg.add_backgroud ('niece', ('Sophia', 'Tomaso'))

 



bg.add_backgroud('grandparent', ('Christopher','Colin'))
bg.add_backgroud('grandparent',('Christopher','Charlotte'))
bg.add_backgroud('grandparent',('Andrew','Colin'))
bg.add_backgroud('grandparent',('Andrew','Charlotte'))
bg.add_backgroud('grandparent',('Roberto', 'Alfonso'))
bg.add_backgroud('grandparent',('Roberto', 'Sophia'))
bg.add_backgroud('grandparent',('Pierro','Alfonso'))
bg.add_backgroud('grandparent',('Pierro','Sophia'))
bg.add_backgroud('grandparent',('Penelope','Colin'))
bg.add_backgroud('grandparent',('Penelope','Charlotte'))
bg.add_backgroud('grandparent',('Christine','Colin'))
bg.add_backgroud('grandparent',('Christine','Charlotte'))
bg.add_backgroud('grandparent',('Maria','Alfonso'))
bg.add_backgroud('grandparent',('Maria','Sophia'))
bg.add_backgroud('grandparent',('Francesca','Alfonso'))
bg.add_backgroud('grandparent',('Francesca','Sophia'))


         
bg.add_all_neg_example('nephew') 

def bgs(it,is_training):
    return [bg,]
 
###########################################################################


parser = argparse.ArgumentParser()

parser.add_argument('--CHECK_CONVERGENCE',default=1,help='Check for convergence',type=int)
parser.add_argument('--SHOW_PRED_DETAILS',default=1,help='Print predicates definition details',type=int)

parser.add_argument('--BS',default=1,help='Batch Size',type=int)
parser.add_argument('--T',default=1 ,help='Number of forward chain',type=int)
parser.add_argument('--LR_SC', default={ (-1000,2):.005 ,  (2,1e5):.003} , help='Learning rate schedule',type=dict)

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
parser.add_argument('--L2',default=0 ,help='Penalty for distance from binary',type=float)
parser.add_argument('--L3',default=0 ,help='Penalty for distance from binary for each term',type=float)
parser.add_argument('--L2LOSS',default=1,help='Use L2 instead of cross entropy',type=int)
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


