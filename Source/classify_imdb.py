from  Lib.ILPRLEngine import *
import argparse
from Lib.mylibw import read_by_tokens
from Lib.DNF import DNF
from Lib.CNF import CNF
from Lib.PredicateLibV5 import PredFunc
from sklearn.metrics import accuracy_score ,precision_recall_curve,auc,precision_recall_fscore_support,average_precision_score,log_loss
from sklearn.metrics import  roc_auc_score ,precision_recall_curve,auc,precision_recall_fscore_support,accuracy_score,confusion_matrix
import pandas as pd
import csv
import operator
import scipy.signal


# for 5-fold we should run the program 5 times with TEST_SET_INDEX from 0 to 4
parser = argparse.ArgumentParser()


parser.add_argument('--TEST_SET_INDEX',default=4,help='0-4 the index of the 5-fold experiment',type=int)
parser.add_argument('--CHECK_CONVERGENCE',default=0,help='Print predicates definition details',type=int)
parser.add_argument('--SHOW_PRED_DETAILS',default=0,help='Print predicates definition details',type=int)
parser.add_argument('--PRINTPRED',default=1,help='Print predicates',type=int)
parser.add_argument('--SYNC',default=0,help='Use L2 instead of cross entropy',type=int)
parser.add_argument('--L2LOSS',default=0,help='Use L2 instead of cross entropy',type=int)
parser.add_argument('--BS',default=1,help='Batch Size',type=int)
parser.add_argument('--T',default=4,help='Number of forward chain',type=int)
parser.add_argument('--LR_SC', default={ (-1000,2):.005 ,  (2,1e5):.05} , help='Learning rate schedule',type=dict)
parser.add_argument('--ITEM_REMOVE_ITER',default=10000 ,help='length period of each item removal',type=int)
parser.add_argument('--MAXTERMS',default=10 ,help='Maximum number of terms in each clause',type=int)
parser.add_argument('--L1',default=0 ,help='Penalty for maxterm',type=float)
parser.add_argument('--L2',default=0 ,help='Penalty for distance from binary',type=float)
parser.add_argument('--L3',default=0 ,help='Penalty for distance from binary',type=float)
parser.add_argument('--ALLTIMESTAMP',default=0 ,help='Add loss for each timestamp',type=int)
parser.add_argument('--LR', default=.003 , help='Base learning rate',type=float)
parser.add_argument('--FILT_TH_MEAN', default=1 , help='Fast convergence total loss threshold MEAN',type=float)
parser.add_argument('--FILT_TH_MAX', default=1 , help='Fast convergence total loss threshold MAX',type=float)
parser.add_argument('--OPT_TH', default=1, help='Per value accuracy threshold',type=float)
parser.add_argument('--PLOGENT', default=.50 , help='Crossentropy coefficient',type=float)
parser.add_argument('--BETA1', default=.90 , help='ADAM Beta1',type=float)
parser.add_argument('--BETA2', default=.999 , help='ADAM Beta2',type=float)
parser.add_argument('--EPS', default=1e-6, help='ADAM Epsillon',type=float)
parser.add_argument('--GPU', default=1, help='Use GPU',type=int)
parser.add_argument('--ITER', default=100000, help='Maximum number of iteration',type=int)
parser.add_argument('--ITER2', default=20, help='Epoch',type=int)
parser.add_argument('--LOGDIR', default='./logs/Logic', help='Log Dir',type=str)
parser.add_argument('--TB', default=0, help='Use Tensorboard',type=int)
parser.add_argument('--ADDGRAPH', default=1, help='Add graph to Tensorboard',type=int)
parser.add_argument('--CLIP_NORM', default=0, help='Clip gradient',type=float)

parser.add_argument('--PRINT_WEIGHTS',default=0,help='Print raw weights',type=int)
parser.add_argument('--SEED',default=0,help='Random seed',type=int)
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
dics=[{} for _ in range(5) ]

for i in range(5):

    IMDB_FILES_I = './data/imdb/imdb.%d.db'%(i+1)
    with open(IMDB_FILES_I, 'r')   as datafile:
        line = datafile.readline()
        while line:
            line=line.replace('(',',')
            line=line.replace(')','')
            line=line.replace('\n','')
            cols = line.split(',')
            cols= [c.strip() for c in cols]
            if cols[0] in dics[i]:
                dics[i][cols[0]].append( cols[1:] )
            else:
                dics[i][cols[0]] = [cols[1:]]
            line = datafile.readline()


#extract the constants from the facts and exaples in the datafiles
################################################

C_person=[ [] for i in range(5) ]
C_movie= [ [] for i in range(5) ]
C_genre= [ [] for i in range(5) ]

for i in range(5):
    for item in dics[i]['director']:
        if item[0] not in C_person[i]:
            C_person[i].append(item[0])
    for item in dics[i]['actor']:
        if item[0] not in C_person[i]:
            C_person[i].append(item[0])
         
    for item in dics[i]['movie']:
        if item[0] not in C_movie[i]:
            C_movie[i].append(item[0])
        if item[1] not in C_person[i]:   
            C_person[i].add(item[1])
    for item in dics[i]['gender']:
        if item[0] not in C_person[i]:
            C_person[i].append(item[0])
    for item in dics[i]['genre']:
        if item[0] not in C_person[i]:
            C_person[i].append(item[0])
        if item[1] not in C_genre[i]:
            C_genre[i].append(item[1])
    for item in dics[i]['workedUnder']:
        if item[0] not in C_person[i]:
            C_person[i].append(item[0])
        if item[1] not in C_person[i]:
            C_person[i].append(item[1])

# we use unified names to be able to generalize across 5 experiments   
#   {p_1,...,p_{nPerson} }   for people names , ....
################################################

nPerson = np.max( [len(C_person[i]) for i in range(5)])
nGenre = np.max( [len(C_genre[i]) for i in range(5)])
nMovie = np.max( [len(C_movie[i]) for i in range(5)])

Persons=[ 'p_%d'%(i+1) for i in range(nPerson)]    
Movies=[ 'm_%d'%(i+1) for i in range(nMovie)]    
Genres=[ 'g_%d'%(i+1) for i in range(nGenre)]    



#defining the predicate functions
################################################
        
Constants = { 'P':Persons, 'G':Genres, 'M':Movies}
predColl = PredCollection (Constants)
 

predColl.add_pred(dname='director',arguments=['P'] )
predColl.add_pred(dname='actor',arguments=['P'] ) 
predColl.add_pred(dname='genre',arguments=['P', 'G'] , variables=[] )
predColl.add_pred(dname='isFemale',arguments=['P'] )
predColl.add_pred(dname='movie',arguments=['M', 'P'] , variables=['P'] )

predColl.add_pred(dname='workedUnder',arguments=['P','P'] , variables=['P','M'] ,pFunc = DNF('workedUnder',terms=1,init=[1,.1,0,.5],sig=1)  , use_neg=True, exc_conds=[('*','rep1') ] , exc_terms=[],  Fam='or', )

# this is the version without the recursion
#predColl.add_pred(dname='workedUnder',arguments=['P','P'] , variables=['P','M'] ,pFunc = DNF('workedUnder',terms=1,init=[1,.1,0,.5],sig=1)  , use_neg=True, exc_conds=[('*','rep1') ] , exc_terms=[],exc_pred=['workedUnder']  Fam='or', )
    

predColl.initialize_predicates()    





def PI( item,i,j):
    return Persons[ C_person[i].index(item[j])] 
def MI( item,i,j):
    return Movies[ C_movie[i].index(item[j])] 
def GI( item,i,j):
    return Genres[ C_genre[i].index(item[j])] 


# defining 5 background knowledge structures corresponding to 5 experiments

all_bgs=[] 

for i in range(5):
    bg = Background( predColl ) 
    
    for item in dics[i]['director']:
        bg.add_backgroud( 'director', ( PI(item,i,0) ,) )

    for item in dics[i]['actor']:
        bg.add_backgroud( 'actor', ( PI(item,i,0) ,) )

    for item in dics[i]['movie']:
        bg.add_backgroud( 'movie', ( MI(item,i,0) , PI(item,i,1) ,) )
        bg.add_example( 'movie', ( MI(item,i,0) , PI(item,i,1) ,) , 1.0)
                

    for item in dics[i]['workedUnder']:
        # since workedUnder is our target we dont add any background facts here
        bg.add_example( 'workedUnder', (  PI(item,i,0)  ,PI(item,i,1) ), 1.0 )

    for item in dics[i]['genre']:
        bg.add_backgroud( 'genre', ( PI(item,i,0) , GI(item,i,1) ,) )
        
    
    for item in dics[i]['gender']:
        bg.add_backgroud( 'isFemale', (  Persons[ C_person[i].index(item[0])]  , ), value = float(item[1]=='Female') )

    
     
    # adding all the negative examples for the target predicate
    bg.add_all_neg_example('workedUnder')    
    
    all_bgs.append(bg)
    
     
 

# this is the callback function that provides the training algorithm with different background knowledge for training and testing
###############################################
def bgs(it,is_train):
     
    if is_train:
        #excluding the background corresponding to the test data
         
        inds = [ i for i in range(5)  if i != args.TEST_SET_INDEX]
        index = np.random.randint(4)
        return [ all_bgs[ inds[ index] ] ]
    else:
        return [ all_bgs[args.TEST_SET_INDEX] ]

    
    
# this callback function is called for custom display each time a testing background is gonna be tested
# ###########################################################################

def disp_fn(eng,it,session,cost,outp):
    
    mismatch_count = 0

    bg=all_bgs[args.TEST_SET_INDEX]

    true_class = bg.get_target_data('workedUnder')
    pred_class = outp['workedUnder'][0,:]>.5
    pred_class = pred_class.astype(np.float)
 
    mask = bg.get_target_mask('workedUnder') 
    true_class = true_class[mask>0]
    pred_class = pred_class[mask>0]
    
    
    auroc = roc_auc_score(true_class, pred_class)
    x,y,th1 = precision_recall_curve(true_class, pred_class)
    aupr = auc(y,x)

    acc = accuracy_score(true_class, (pred_class>=.5).astype(np.float) )
    
    print('--------------------------------------------------------')    
    print('acc : %.4f , AUROC : %.4f,  AUPR : %.4f'% (acc,auroc,aupr) )

    


model = ILPRLEngine( args=args ,predColl=predColl ,bgs=bgs ,disp_fn=disp_fn)
model.train_model()    


