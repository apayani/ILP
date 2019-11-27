
from  Lib.ILPRLEngine import *
import argparse
from Lib.mylibw import read_by_tokens
from Lib.DNF import DNF
from Lib.CONJ import CONJ
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
    parser.add_argument('--TEST_SET_INDEX',default=3,help='0-9 the index of the 10-fold experiment',type=int)
    parser.add_argument('--CHECK_CONVERGENCE',default=0,help='Print predicates definition details',type=int)
    parser.add_argument('--SHOW_PRED_DETAILS',default=0,help='Print predicates definition details',type=int)
    parser.add_argument('--PRINTPRED',default=0,help='Print predicates',type=int)
    parser.add_argument('--SYNC',default=0, help='Use L2 instead of cross entropy',type=int)
    parser.add_argument('--L2LOSS',default=0,help='Use L2 instead of cross entropy',type=int)
    parser.add_argument('--BS',default=40,help='Batch Size',type=int)
    parser.add_argument('--T',default=1,help='Number of forward chain',type=int)
    parser.add_argument('--LR_SC', default={ (-1000,10):.05, (8,10):.05,  (10,1e5):.05} , help='Learning rate schedule',type=dict)
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
    parser.add_argument('--ITER', default=500, help='Maximum number of iteration',type=int)
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



u2base=pd.read_csv( './data/MovieLens/u2base.csv')
actors=pd.read_csv( './data/MovieLens/actors.csv')
directors=pd.read_csv( './data/MovieLens/directors.csv')
users=pd.read_csv( './data/MovieLens/users.csv')
movies=pd.read_csv( './data/MovieLens/movies.csv')
movies2actors=pd.read_csv( './data/MovieLens/movies2actors.csv')
movies2directors=pd.read_csv( './data/MovieLens/movies2directors.csv')


u2basej = u2base.join( movies2directors.set_index('movieid'), on='movieid', lsuffix='', rsuffix='_other')
u2basej = u2basej.join( movies.set_index('movieid'), on='movieid', lsuffix='', rsuffix='_other')
u2basej = u2basej.join( directors.set_index('directorid') , on='directorid', lsuffix='', rsuffix='_other' )
 

 

#defining the predicate functions
################################################

MAX_MOVIE = 100


MOVIES=[ '%d'%(i) for i in range(MAX_MOVIE)]
GENRES = list( movies2directors['genre'].unique() )
RATINGS = list( u2base['rating'].unique() )
AGES = list( users['age'].unique() )
OCCUPATION = list( users['occupation'].unique() )

# Molecules=[ '%d'%(i+1) for i in range(MAX_Molecules)]


Constants = { 'M':MOVIES,'N':['%d'%i for i in range(6)]}
predColl = PredCollection (Constants)
 
for a in AGES:
    predColl.add_pred( dname='age_%d'%a,arguments=[])

for a in AGES:
    predColl.add_pred( dname='olderthan_%d'%a,arguments=[])


for a in OCCUPATION:
    predColl.add_pred( dname='occupation_%d'%a,arguments=[])

for r in RATINGS:
    predColl.add_pred( dname='rate_%d'%r,arguments=['M'])


for g in GENRES:
    predColl.add_pred(dname='genre_'+g,arguments=['M']  )


for i in list( u2basej['d_quality'].unique()) :
    predColl.add_pred(dname='d_quality_'+str(i),arguments=['M']  )

for i in list( u2basej['avg_revenue'].unique()) :
    predColl.add_pred(dname='avg_revenue_'+str(i),arguments=['M']  )
    
for i in list( u2basej['isEnglish'].unique()) :
    predColl.add_pred(dname='isEnglish_'+str(i),arguments=['M']  )

for i in [ '%d'%(i) for i in range(5)]:
    predColl.add_pred(dname='movie_year_'+str(i),arguments=['M']  )

for i in range(10):

    predColl.add_pred(dname='female%d'%i,oname='female',arguments=[], variables=['M' ] ,pFunc = 
        DNF('female%d'%i,terms=4,init=[-1,.1,-1,.1],sig=2 ,predColl=predColl)  , use_neg=True,  Fam='or',chunk_count=0 ) 
    

predColl.initialize_predicates()    



 

# extracting bacground knowledge corresponding to each user
bgs_pos=[] 
bgs_neg=[] 

for i,u in users.iterrows():
    bg = Background( predColl ) 

    
    bg.add_backgroud( 'occupation_%s'%(str(u['occupation'])) , ()  )
    
    bg.add_backgroud( 'age_%s'%(str(u['age'])) , ()  )
    for a in AGES:
        if u['age']>a:
            bg.add_backgroud( 'olderthan_%s'%(str(u['age'])) , ()  )
    
    
    bg.add_example('female', (), value= float( u['u_gender']=='M') )
    
    df_mask = u2basej['userid']==u['userid']
    df = u2basej[df_mask]


    movie_cnt=0
    for j, row_rating in df.iterrows():
        
        
        rate=row_rating['rating']    
        
        movieid  =row_rating['movieid']
        dfm=movies2directors[movies2directors['movieid'] == movieid]
        if dfm['genre'].size<1:
             continue
        
        genre= row_rating['genre'] 
        if genre!=genre:
            continue
        
        
        bg.add_backgroud( 'genre_'+genre, ('%d'%(movie_cnt),) )
        bg.add_backgroud( 'd_quality_'+str(row_rating['d_quality']), ('%d'%(movie_cnt), ) )
        bg.add_backgroud( 'avg_revenue_'+str(row_rating['avg_revenue']), ('%d'%(movie_cnt), ) )
        bg.add_backgroud( 'isEnglish_'+str(row_rating['isEnglish']), ('%d'%(movie_cnt), ) )
        bg.add_backgroud( 'rate_%d'%rate, ('%d'%(movie_cnt),) )
         
         

        movie_cnt+=1
        if movie_cnt>=MAX_MOVIE:
            break

    if i%100==0:
        print('%d of total %d users are loaded'%(i,users.shape[0]))
     
    if u['u_gender']=='F':
        bgs_pos.append(bg)
    else:
        bgs_neg.append(bg)

print('finished loading background knowledge')
np.random.seed(0)

#creating 5 fold data 
inds_pos=np.arange(len(bgs_pos))
inds_neg=np.arange(len(bgs_neg))
kf = KFold(5,True,args.SEED)
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


print(len(train_data))
print(len(test_data))
 
 
    
     
def bgs(it,is_train):
     
    if is_train:
        inds_pos=np.random.permutation(len(train_data))
        return [ train_data[inds_pos[i]]  for i in range(500)] 
    else:
        return test_data


    
    


def disp_fn(eng,it,session,cost,outp):
    


    o = outp['female']
    
    true_class = []
    pred_class = []
    
    for i in range(len(o)):
    
        true_class.append(    float( test_data[i].get_target_data('female') ) )
        pred_class.append(  float( o[i][0]) )
    

    true_class = np.array(true_class)
    pred_class = np.array(pred_class)
    
    

    
    
    avg_acc = average_precision_score(true_class, pred_class)
    auroc = roc_auc_score(true_class, pred_class)
    x,y,th1 = precision_recall_curve(true_class, pred_class)
    aupr = auc(y,x)
 
 
    print('--------------------------------------------------------')    
    print('acc : %.4f,  AUROC : %.4f,  AUPR : %.4f'% (acc,auroc,aupr ) )
 
    return

     

model = ILPRLEngine( args=args ,predColl=predColl ,bgs=bgs ,disp_fn=disp_fn)
model.train_model()    


