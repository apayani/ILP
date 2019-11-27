from  Lib.ILPRLEngine import *
import argparse
from Lib.mylibw import read_by_tokens
from Lib.DNF import DNF
from Lib.CNF import CNF
from Lib.CONJ  import CONJ
from Lib.cgt import CGT


from Lib.PredicateLibV5 import PredFunc
from sklearn.metrics import accuracy_score ,precision_recall_curve,auc,precision_recall_fscore_support,average_precision_score,log_loss
from sklearn.metrics import  roc_auc_score ,precision_recall_curve,auc,precision_recall_fscore_support,accuracy_score,confusion_matrix
import pandas as pd
import csv
import operator
import scipy.signal


# for 5-fold we should run the program 5 times with TEST_SET_INDEX from 0 to 4
parser = argparse.ArgumentParser()


parser.add_argument('--TEST_SET_INDEX',default=2,help='0-4 the index of the 5-fold experiment',type=int)
parser.add_argument('--CHECK_CONVERGENCE',default=0,help='Print predicates definition details',type=int)
parser.add_argument('--SHOW_PRED_DETAILS',default=0,help='Print predicates definition details',type=int)
parser.add_argument('--PRINTPRED',default=0,help='Print predicates',type=int)
parser.add_argument('--SYNC',default=0,help='Use L2 instead of cross entropy',type=int)
parser.add_argument('--L2LOSS',default=0,help='Use L2 instead of cross entropy',type=int)
parser.add_argument('--BS',default=1,help='Batch Size',type=int)
parser.add_argument('--T',default=4,help='Number of forward chain',type=int)
parser.add_argument('--LR_SC', default={ (-1000,2):.01 ,  (2,1e5):.1} , help='Learning rate schedule',type=dict)
parser.add_argument('--ITEM_REMOVE_ITER',default=10000 ,help='length period of each item removal',type=int)
parser.add_argument('--MAXTERMS',default=6 ,help='Maximum number of terms in each clause',type=int)
parser.add_argument('--L1',default=0 ,help='Penalty for maxterm',type=float)
parser.add_argument('--L2',default=.0 ,help='Penalty for distance from binary',type=float)
parser.add_argument('--L3',default=0 ,help='Penalty for distance from binary',type=float)
parser.add_argument('--ALLTIMESTAMP',default=0 ,help='Add loss for each timestamp',type=int)
parser.add_argument('--LR', default=.003 , help='Base learning rate',type=float)
parser.add_argument('--FILT_TH_MEAN', default=1 , help='Fast convergence total loss threshold MEAN',type=float)
parser.add_argument('--FILT_TH_MAX', default=1 , help='Fast convergence total loss threshold MAX',type=float)
parser.add_argument('--OPT_TH', default=1, help='Per value accuracy threshold',type=float)
parser.add_argument('--PLOGENT', default=.850 , help='Crossentropy coefficient',type=float)
parser.add_argument('--BETA1', default=.90 , help='ADAM Beta1',type=float)
parser.add_argument('--BETA2', default=.999 , help='ADAM Beta2',type=float)
parser.add_argument('--EPS', default=1e-6, help='ADAM Epsillon',type=float)
parser.add_argument('--GPU', default=1, help='Use GPU',type=int)
parser.add_argument('--ITER', default=4800, help='Maximum number of iteration',type=int)
parser.add_argument('--ITER2', default=10, help='Epoch',type=int)
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
dics_neg=[{} for _ in range(5) ]
names=[ '1.db','2.db','3.db','4.db','5.db']
for i in range(5):

    DATA_FILES_I = './data/cora/coraSepFixedhwng.'  + names[i]
    with open(DATA_FILES_I, 'r')   as datafile:
        line = datafile.readline()
        while line:
            line=line.replace('(',',')
            line=line.replace(')','')
            line=line.replace('\n','')
            cols = line.split(',')
            cols= [c.strip() for c in cols]
            neg=False
            if cols[0].startswith('!'):
                cols[0]=cols[0][1:]
                neg = True
            

            if cols[0] in dics[i]:
                dics[i][cols[0]].append( cols[1:] )
                dics_neg[i][cols[0]].append( neg )
            else:
                dics[i][cols[0]] = [cols[1:]]
                dics_neg[i][cols[0]] = [neg]

            
            line = datafile.readline()


#extract the constants from the facts and exaples in the datafiles
################################################

C_author=[ [] for i in range(5) ]
C_class= [ [] for i in range(5) ]
C_title=[ [] for i in range(5) ]
C_venue=[ [] for i in range(5) ]
C_word=[ [] for i in range(5) ]

 
cnt=0

for i in range(5):
    for item in dics[i]['Author']:
        if item[0] not in C_class[i]:
            C_class[i].append(item[0])
        if item[1] not in C_author[i]:
            C_author[i].append(item[1])


    for item in dics[i]['Title']:
        if item[0] not in C_class[i]:
            C_class[i].append(item[0])
        if item[1] not in C_title[i]:
            C_title[i].append(item[1])
        
    for item in dics[i]['Venue']:
        if item[0] not in C_class[i]:
            C_class[i].append(item[0])
        if item[1] not in C_venue[i]:
            C_venue[i].append(item[1])

    for item in dics[i]['HasWordAuthor']:
        if item[0] not in C_author[i]:
            C_author[i].append(item[0])
        if item[1] not in C_word[i]:
            C_word[i].append(item[1])
    
    for item in dics[i]['HasWordTitle']:
        if item[0] not in C_title[i]:
            C_title[i].append(item[0])
        if item[1] not in C_word[i]:
            C_word[i].append(item[1])

    for item in dics[i]['HasWordVenue']:
        if item[0] not in C_venue[i]:
            C_venue[i].append(item[0])
        if item[1] not in C_word[i]:
            C_word[i].append(item[1])

    for item in dics[i]['SameBib']:
        if item[0] not in C_class[i]:
            C_class[i].append(item[0])
        if item[1] not in C_class[i]:
            C_class[i].append(item[1])
    
    for item in dics[i]['SameAuthor']:
        if item[0] not in C_author[i]:
            C_author[i].append(item[0])
        if item[1] not in C_author[i]:
            C_author[i].append(item[1])

    for item in dics[i]['SameTitle']:
        if item[0] not in C_title[i]:
            C_title[i].append(item[0])
        if item[1] not in C_title[i]:
            C_title[i].append(item[1])

    for item in dics[i]['SameVenue']:
        if item[0] not in C_venue[i]:
            C_venue[i].append(item[0])
        if item[1] not in C_venue[i]:
            C_venue[i].append(item[1])
    

 ###############################################

nAuthor = np.max( [len(C_author[i]) for i in range(5)])
nClass = np.max( [len(C_class[i]) for i in range(5)])
nTitle = np.max( [len(C_title[i]) for i in range(5)])
nVenue = np.max( [len(C_venue[i]) for i in range(5)])
nWord = np.max( [len(C_word[i]) for i in range(5)])


Authors=[ 'a_%d'%(i+1) for i in range(nAuthor)]    
Classes=[ 'c_%d'%(i+1) for i in range(nClass)]    
Titles=[ 't_%d'%(i+1) for i in range(nTitle)]    
Venues=[ 'v_%d'%(i+1) for i in range(nVenue)]    
Words=[ 'w_%d'%(i+1) for i in range(nWord)]    

def AI( item,i):
    return Authors[ C_author[i].index(item)] 
def CI( item,i):
    return Classes[ C_class[i].index(item)] 
def TI( item,i):
    return Titles[ C_title[i].index(item)] 
def VI( item,i):
    return Venues[ C_venue[i].index(item)] 

def WI( item,i):
    return Words[ C_word[i].index(item)] 

pair_author=[]
for i in range(5):
    for item in dics[i]['Author']:
        pair_author.append (  (CI(item[0],i),  AI(item[1],i) ) )
        
pair_title=[]
for i in range(5):
    for item in dics[i]['Title']:
        pair_title.append (  (CI(item[0],i),  TI(item[1],i) ) )

pair_venue=[]
for i in range(5):
    for item in dics[i]['Venue']:
        pair_venue.append (  (CI(item[0],i),  VI(item[1],i) ) )


pair_HasWordAuthor=[]
for i in range(5):
    for item in dics[i]['HasWordAuthor']:
        pair_HasWordAuthor.append (  (AI(item[0],i),  WI(item[1],i) ) )

pair_HasWordTitle=[]
for i in range(5):
    for item in dics[i]['HasWordTitle']:
        pair_HasWordTitle.append (  (TI(item[0],i),  WI(item[1],i) ) )

pair_HasWordVenue=[]
for i in range(5):
    for item in dics[i]['HasWordVenue']:
        pair_HasWordVenue.append (  (VI(item[0],i),  WI(item[1],i) ) )

pair_SameAuthor=[]
for i in range(5):
    for item in dics[i]['SameAuthor']:
        pair_SameAuthor.append (  (AI(item[0],i),  AI(item[1],i) ) )


pair_SameVenue=[]
for i in range(5):
    for item in dics[i]['SameVenue']:
        pair_SameVenue.append (  (VI(item[0],i),  VI(item[1],i) ) )


pair_SameTitle=[]
for i in range(5):
    for item in dics[i]['SameTitle']:
        pair_SameTitle.append (  (TI(item[0],i),  TI(item[1],i) ) )


pair_SameBib=[]
for i in range(5):
    for j,item in enumerate( dics[i]['SameBib']):
        # if not dics_neg[i]['SameBib'][j] :
        pair_SameBib.append (  (CI(item[0],i),  CI(item[1],i) ) )

for c in Classes:
    pair_SameBib.append( (c,c))
for v in Venues:
    pair_SameVenue.append( (v,v))
for t in Titles:
    pair_SameTitle.append( (t,t))
for a in Authors:
    pair_SameAuthor.append( (a,a))

pair_SameBib = list( set (pair_SameBib))
pair_SameVenue = list( set (pair_SameVenue))
pair_SameTitle = list( set (pair_SameTitle))
pair_SameAuthor = list( set (pair_SameAuthor))
#defining the predicate functions
################################################
        
Constants = { 'A':Authors, 'C':Classes, 'T':Titles, 'V':Venues, 'W':Words}

predColl = PredCollection (Constants)
 



predColl.add_pred(dname='HasWordAuthor',arguments=['A','W'],pairs=pair_HasWordAuthor)
predColl.add_pred(dname='HasWordTitle',arguments=['T','W'],pairs=pair_HasWordTitle)
predColl.add_pred(dname='HasWordVenue',arguments=['V','W'],pairs=pair_HasWordVenue)
 

predColl.add_pred(dname='SameAuthor',arguments=['A','A'],pairs=pair_SameAuthor)
predColl.add_pred(dname='SameTitle',arguments=['T','T'],pairs=pair_SameTitle)
# predColl.add_pred(dname='SameVenue',arguments=['V','V'],pairs=pair_SameVenue)



predColl.add_pred(dname='Author',arguments=['C','A'], variables=['A'], pairs=pair_author,pFunc = 
    CONJ('Author',init=[-1,.1],sig=2,init_terms=['SameAuthor(B,C), Author(A,C)'],predColl=predColl,fast=True)  , use_neg=False, inc_preds=['SameAuthor','Author'], exc_conds=[ ],  Fam='or',chunk_count=0) 

predColl.add_pred(dname='Title',arguments=['C','T'], variables=['T'], pairs=pair_title,pFunc = 
    CONJ('Title',init=[-1,.1],sig=2,init_terms=['SameTitle(B,C), Title(A,C)'],predColl=predColl,fast=True)  , use_neg=False, inc_preds=['SameTitle','Title'], exc_conds=[ ],  Fam='or',chunk_count=0) 

# predColl.add_pred(dname='Title',arguments=['C','T'],pairs=pair_title)
predColl.add_pred(dname='Venue',arguments=['C','V'],pairs=pair_venue)

predColl.add_pred(dname='sameWTA',arguments=['T','A'], variables=['W'] ,pFunc = 
    CONJ('sameWTA',init=[-1,.1],sig=2,init_terms=['HasWordTitle(A,C), HasWordAuthor(B,C)'],predColl=predColl,fast=True)  , use_neg=False, inc_preds=['HasWordTitle','HasWordAuthor'], exc_conds=[ ],  Fam='or',chunk_count=0) 

# predColl.add_pred(dname='sameWTV',arguments=['T','V'], variables=['W'] ,pFunc = 
#     CONJ('sameWTV',init=[-1,.1],sig=2,init_terms=['HasWordTitle(A,C), HasWordVenue(B,C)'],predColl=predColl,fast=True)  , use_neg=False, inc_preds=['HasWordTitle','HasWordVenue'], exc_conds=[ ],  Fam='or',chunk_count=0) 

# predColl.add_pred(dname='sameWAV',arguments=['A','V'], variables=['W'] ,pFunc = 
#     CONJ('sameWAV',init=[-1,.1],sig=2,init_terms=['HasWordAuthor(A,C), HasWordVenue(B,C)'],predColl=predColl,fast=True)  , use_neg=False, inc_preds=['HasWordAuthor','HasWordVenue'], exc_conds=[ ],  Fam='or',chunk_count=0) 

predColl.add_pred(dname='SameBib1',oname='SameBib',arguments=['C','C'], variables=[  'A' ] ,pFunc = DNF('SameBib1',terms=1,init=[-1,.1,-1,.1],sig=2)  , use_neg=True , Fam='or',chunk_count=0,pairs=pair_SameBib)
predColl.add_pred(dname='SameBib2',oname='SameBib',arguments=['C','C'], variables=[  'T' ] ,pFunc = DNF('SameBib2',terms=1,init=[-1,.1,-1,.1],sig=2)  , use_neg=True , Fam='or',chunk_count=0,pairs=pair_SameBib)
    

predColl.initialize_predicates()    



# defining 5 background knowledge structures corresponding to 5 experiments

all_bgs=[] 

for i in range(5):
    bg = Background( predColl ) 
    
    for j,item in enumerate(dics[i]['Author']):
        try:
            if not dics_neg[i]['Author'][j]:
                bg.add_backgroud( 'Author', ( CI(item[0],i),AI(item[1],i) ,) )
        except:
            print(i,item,'Author')


    for j,item in enumerate(dics[i]['Title']):
        try:
            if not dics_neg[i]['Title'][j]:
                bg.add_backgroud( 'Title', ( CI(item[0],i),TI(item[1],i) ,) )
        except:
            print(i,item,'Title')

    
    for j,item in enumerate(dics[i]['Venue']):
        try:
            if not dics_neg[i]['Venue'][j]:
                bg.add_backgroud( 'Venue', ( CI(item[0],i),VI(item[1],i) ,) )
        except:
            print(i,item,'Venue')


    for j,item in enumerate(dics[i]['HasWordAuthor']):
        try:
            if not dics_neg[i]['HasWordAuthor'][j]:
                bg.add_backgroud( 'HasWordAuthor', ( AI(item[0],i),WI(item[1],i) ,) )
        except:
            print(i,item,'Author')

    for j,item in enumerate(dics[i]['HasWordTitle']):
        try:
            if not dics_neg[i]['HasWordTitle'][j]:
                bg.add_backgroud( 'HasWordTitle', ( TI(item[0],i),WI(item[1],i) ,) )
        except:
            print(i,item,'HasWordTitle')

    for j,item in enumerate(dics[i]['HasWordVenue']):
        try:
            if not dics_neg[i]['HasWordVenue'][j]:
                bg.add_backgroud( 'HasWordVenue', ( VI(item[0],i),WI(item[1],i) ,) )
        except:
            print(i,item,'HasWordVenue')


    

    for j,item in enumerate(dics[i]['SameTitle']):
        try:
            if not dics_neg[i]['SameTitle'][j]:
                bg.add_backgroud( 'SameTitle', ( TI(item[0],i),TI(item[1],i) ,) )
        except:
            pass
            # print(i,item,'SameTitle')

    for j,item in enumerate(dics[i]['SameVenue']):
        try:
            if not dics_neg[i]['SameVenue'][j]:
                bg.add_backgroud( 'SameVenue', ( VI(item[0],i),VI(item[1],i) ,) )
        except:
            pass

    for j,item in enumerate(dics[i]['SameAuthor']):
        try:
            if not dics_neg[i]['SameAuthor'][j]:
                bg.add_backgroud( 'SameAuthor', ( AI(item[0],i),AI(item[1],i) ,) )
        except:
            pass
            # print(i,item,'SameAuthor')

    for j,item in enumerate(dics[i]['SameBib']):
        try:
            
            bg.add_example( 'SameBib', ( CI(item[0],i),CI(item[1],i) ,),value=float(not dics_neg[i]['SameBib'][j])  )
        except:
            pass
    
    
    
    all_bgs.append(bg)
    
     
 

# this is the callback function that provides the training algorithm with different background knowledge for training and testing
###############################################
def bgs(it,is_train):
     
    if is_train:


        inds = [ i for i in range(5)  if i != args.TEST_SET_INDEX]
        return [ all_bgs[ inds[it%4] ] ]

    else:

        return [ all_bgs[args.TEST_SET_INDEX] ]

    
    
# this callback function is called for custom display each time a testing background is gonna be tested
# ###########################################################################

def disp_fn(eng,it,session,cost,outp):
    
    mismatch_count = 0

    bg=all_bgs[args.TEST_SET_INDEX]

    mask = bg.get_target_mask('SameBib')
    true_class = bg.get_target_data('SameBib')
    pred_class =outp['SameBib'][0,:]

    true_class = true_class[mask>0]
    pred_class = pred_class[mask>0]
    
   
    auroc = roc_auc_score(true_class, pred_class)
    x,y,th1 = precision_recall_curve(true_class, pred_class)
    aupr = auc(y,x)


    acc = accuracy_score(true_class, (pred_class>=.5).astype(np.float) )
    
    print('--------------------------------------------------------')    
    print('acc : %.4f , AUROC : %.4f,  AUPR : %.4f'% (acc,auroc,aupr) )

    
 
    return

     

model = ILPRLEngine( args=args ,predColl=predColl ,bgs=bgs ,disp_fn=disp_fn)
model.train_model()    

