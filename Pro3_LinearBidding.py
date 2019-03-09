# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 21:51:54 2019

@author: 白开心
"""
#%%
import numpy as np
import pandas as pd
import seaborn as sb
import sklearn
import re
import os
import matplotlib.pyplot as plt

from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
# from sklearn.cross_validation import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report

%matplotlib inline
rcParams['figure.figsize'] = 10, 8
sb.set_style('whitegrid')
#%%
train_file = pd.read_csv('train.csv') # train.csv
#%% negative down sampling
def neg_sampling (df, w):
    df_click = df[df['click'] == 1]
    df_noclick = df[df['click'] == 0]
    df_noclick_sample = df_noclick.sample(frac = w, axis = 0)  # random_state = int
    df_sample = df_click.append(df_noclick_sample).reset_index(drop = True)
    return df_sample
frac = 0.15
train = neg_sampling(train_file, frac)
#%% drop irrelevant variables
train = train.drop(['bidid','userid','IP','domain','url','urlid','slotid','bidprice','payprice','keypage'], 1)
train_label = train['click']
train = train.drop(['click'], 1)
train.head()
#%% check if target variable is binary
train_label.value_counts()
#%% check for missing values
train.isnull().sum()
#%% conver numerical data into categories
sorted(train['slotwidth'].unique())
bin = [-1,200,400,600,800,max(train['slotwidth'])]
slotwidth = pd.cut(train['slotwidth'],bin)
slotwidth = slotwidth.to_frame()

sorted(train['slotheight'].unique())
bin = [-1,100,200,300,400,500,max(train['slotwidth'])]
slotheight = pd.cut(train['slotheight'],bin)
slotheight = slotheight.to_frame()

sorted(train['slotprice'].unique())
bin = [-1,50,100,150,200,250,max(train['slotprice'])]
slotprice = pd.cut(train['slotprice'],bin)
slotprice = slotprice.to_frame()

train = train.drop(['slotwidth','slotheight','slotprice'], 1)
train = pd.concat([train,slotwidth,slotheight,slotprice],axis = 1)
train.dtypes
#%% clean 'usertag'
usertag = train['usertag'].astype(str)
for index in range(0,len(usertag)):
    if usertag[index] != 'nan':
        usertag[index] = int(re.sub(',', '', usertag[index]))
        if usertag[index] > 100000000000000000000:
            usertag[index] = 'ut0'
train = train.drop(['usertag'], 1)
train = pd.concat([train,usertag],axis = 1)
train.dtypes
#%% convert numerical data into str
num_cat = ['weekday','hour','region','city','adexchange','advertiser']
for var in num_cat:
    train[var] = train[var].apply(str)
train.dtypes
#%%
validation = pd.read_csv('validation.csv') # validation.csv
validation = validation.drop(['bidid','userid','IP','domain','url','urlid','slotid','bidprice','payprice','keypage'], 1)
validation_label = validation['click']
validation = validation.drop(['click'], 1)

bin = [-1,200,400,600,800,max(validation['slotwidth'])]
slotwidth = pd.cut(validation['slotwidth'],bin)
slotwidth = slotwidth.to_frame()
bin = [-1,100,200,300,400,500,max(validation['slotwidth'])]
slotheight = pd.cut(validation['slotheight'],bin)
slotheight = slotheight.to_frame()
bin = [-1,50,100,150,200,250,max(validation['slotprice'])]
slotprice = pd.cut(validation['slotprice'],bin)
slotprice = slotprice.to_frame()
validation = validation.drop(['slotwidth','slotheight','slotprice'], 1)
validation = pd.concat([validation,slotwidth,slotheight,slotprice],axis = 1)

usertag = validation['usertag'].astype(str)
for index in range(0,len(usertag)):
    if usertag[index] != 'nan':
        usertag[index] = int(re.sub(',', '', usertag[index]))
        if usertag[index] > 100000000000000000000:
            usertag[index] = 'ut0'
validation = validation.drop(['usertag'], 1)
validation = pd.concat([validation,usertag],axis = 1)

num_cat = ['weekday','hour','region','city','adexchange','advertiser']
for var in num_cat:
    validation[var] = validation[var].apply(str)
#%% convert into one-hot, train LR model
train_valid_together = pd.concat([train,validation],axis=0)
together_oh = pd.get_dummies(train_valid_together,dummy_na=True)
train_oh = together_oh[0:len(train)]
validation_oh = together_oh[len(train):len(together_oh)]

#train_oh.to_csv('train_oh.csv')
#train_label.to_csv('train_label.csv')
#validation_oh.to_csv('validation_oh.csv')
#validation_label.to_csv('validation_label.csv')

del together_oh
del train_valid_together
del train_file
del slotheight
del slotprice
del slotwidth
del usertag

LogReg = LogisticRegression(solver='liblinear')
LogReg.fit(train_oh, train_label)
#%% compute pCTR for valiadation set
pctr_1 = LogReg.predict_proba(validation_oh)[:, 1]  # before re-calibration
pctr_2 = frac/(frac-1+1/pctr_1)  # after re-calibration
validation_label = validation['click'] 
pred_ctr_3 = pred_ctr_2.copy()  # convert into binary value
pctr_3[pctr_3 >= 0.5] = 1
pctr_3[pctr_3 < 0.5] = 0

pd_pctr_validation = pd.DataFrame(pctr_2,columns=['pctr'])
print(pd_pctr_validation)
pd_pctr_validation.to_csv('../validation_pctr.csv')
#%% estimation
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(validation_label, pctr_3)
print(classification_report(validation_label, pctr_3))
#%% ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(validation_label, pctr_3)
fpr, tpr, thresholds = roc_curve(validation_label, pctr_2)
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.5f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('pCTR_ROC')
plt.show()
#%% base_bid optimization
#%% bid = base_bid * pCTR / avgCTR
validation_pctr = pd.read_csv('../validation_pctr.csv')
avgCTR = 1793/2430981
budge = 6250000
record = np.zeros((300,4),dtype = np.float)

i = 0
for base_bid in range(1,301):
    cost = 0
    impression = 0
    click = 0
    for index in validation_pctr.index:  
        bid = base_bid*validation_pctr['pCTR'][index]/avgCTR  # generate a bid
        if bid >= validation['payprice'][index]:
            impression = impression+1
            cost = cost+validation['payprice'][index]
            if validation_label[index] == 1:
                click = click+1
    record[i] = [base_bid,impression,click,cost]
    i = i+1
#%%    
pd_record = pd.DataFrame(record,columns=['base_bid','num_impression','num_click','cost'])
print(pd_record)
pd_record.to_csv('../basebid_0_300.csv')
#%% optimal base bid
base_bid = 82
click = 133
impression = 132916
cost = 6250000

# avg.cpm = budge/num_impre*1000
# avg.cpc = cost per click = budge/1000/num_click
ctr = click/impression
avg_cpm = cost/impression*1000
avg_cpc = budge/1000/click

print(click)
print(impression)
print(cost)
print(ctr)
print(avg_cpm)
print(avg_cpc)
#%% calculate bidprice for test dataset
#%%
test = pd.read_csv('test.csv') # test.csv
test = test.drop(['bidid','userid','IP','domain','url','urlid','slotid','keypage'], 1)

bin = [-1,200,400,600,800,max(test['slotwidth'])]
slotwidth = pd.cut(test['slotwidth'],bin)
slotwidth = slotwidth.to_frame()
bin = [-1,100,200,300,400,500,max(test['slotwidth'])]
slotheight = pd.cut(test['slotheight'],bin)
slotheight = slotheight.to_frame()
bin = [-1,50,100,150,200,250,max(test['slotprice'])]
slotprice = pd.cut(test['slotprice'],bin)
slotprice = slotprice.to_frame()
test = test.drop(['slotwidth','slotheight','slotprice'], 1)
test = pd.concat([test,slotwidth,slotheight,slotprice],axis = 1)

usertag = test['usertag'].astype(str)
for index in range(0,len(usertag)):
    if usertag[index] != 'nan':
        usertag[index] = int(re.sub(',', '', usertag[index]))
        if usertag[index] > 100000000000000000000:
            usertag[index] = 'ut0'
test = test.drop(['usertag'], 1)
test = pd.concat([test,usertag],axis = 1)

num_cat = ['weekday','hour','region','city','adexchange','advertiser']
for var in num_cat:
    test[var] = test[var].apply(str)
#%% convert into one-hot, train LR model
train_test_together = pd.concat([train,test],axis=0)
together_oh = pd.get_dummies(train_test_together,dummy_na=True)
train_oh = together_oh[0:len(train)]
test_oh = together_oh[len(train):len(together_oh)]

LogReg = LogisticRegression(solver='liblinear')
LogReg.fit(train_oh, train_label)
#%% compute pCTR for valiadation set
pctr_4 = LogReg.predict_proba(test_oh)[:, 1]  # before re-calibration
pctr_5 = frac/(frac-1+1/pctr_1)  # after re-calibration

pd_pctr_test = pd.DataFrame(pctr_5,columns=['pctr'])
print(pd_pctr_test)
pd_pctr_test.to_csv('../test_pctr.csv')
#%% calculate bidprice for test set
test_pctr = pd.read_csv('test_pctr.csv')
bidprice = np.zeros((len(test_pctr),1),dtype = np.float)
for index in test_pctr.index:
    bidprice[index] = base_bid*test_pctr['pCTR'][index]/avgCTR
#%%
pd_bidprice = pd.DataFrame(bidprice,columns=['bidprice',])
print(pd_bidprice)
pd_bidprice.to_csv('../linear_bidprice_82.csv')    

    
    
    
    
    
    
    