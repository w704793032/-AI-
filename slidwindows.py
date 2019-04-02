# return size+1 size*fetures & 1*result
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import os

creatVars = locals()


def slidwindows(dt,size):
    daylist = np.sort(dt['day'].unique())
    if size>len(daylist):
        print('window size out of range.')
        return 
    
    for i in range(0,len(daylist)-size): 
        #获取n+1天相关数据  'stationID','lineID','week','day','hour','minute','inNums','outNums'
        item_day = daylist[i+size]
        batch_data = dt[dt['day']==item_day][['stationID','lineID','week','day','hour','minute','inNums','outNums']]
        #获取前n天数据
        for j in range(1,size+1):
            tmp = dt[dt['day']==(item_day-j)]
            cols = tmp.columns.drop(['stationID','lineID','hour','minute'])
            tmp.drop(['week','day'],axis=1,inplace=True)
          #  tmp = tmp[['stationID','lineID','hour','minute','inNums','outNums']]
            for f in cols:
                tmp.rename(columns={f: f+'_last_'+str(j)}, inplace=True)
            batch_data = pd.merge(batch_data,tmp,on=['stationID','lineID','hour','minute'],how='left')
        if i ==0:
            res_data = batch_data
        else:
            res_data = pd.concat([res_data,batch_data])
    return res_data
	
def pltinoutsta(df,plot=True):
    #统计每一个十分钟时间段，进出站的人数
    insta=[]
    outsta=[]
    minsteps=df['minstep'].unique()
    for minstep in minsteps:
        insta.append((df[df['minstep']==minstep]['status']==0).sum())
        outsta.append((df[df['minstep']==minstep]['status']==1).sum())
    if plot==True:
        plt.plot(minsteps,insta,color='red',label='insta')
        plt.plot(minsteps,outsta,color='green',label='outsta')
        plt.show()
    else:
        pass
    return minsteps,insta,outsta	
	
def MAE(value1,value2):
    return np.sum(np.abs(value1-value2))/len(value1)

def totaldata():
    data = pd.read_csv('../input/Metro_testA/testA_record_2019-01-28.csv')
    print(f'data 28,date 2019-01-28')  
    for i in range(1,26):
        if i<10:
            s = str(0)+str(i)
        else:
            s = str(i)    
       # print(f'data {i},date 2019-01-{s}')      
        traindata_batch = pd.read_csv('../input/Metro_train/record_2019-01-'+s+'.csv')
        data = pd.concat([data,traindata_batch])
        del traindata_batch
        gc.collect()
    return data
def findfreqcaller(data,threshold):
    counts = data[data['payType']!=3]['userID'].value_counts()
    return list(counts[counts>threshold].index)

def statis_feature(dt):
    tmp = dt.groupby(['stationID','week','hour','minute'], as_index=False)['inNums'].agg({
                                                                            'inNums_whm_max'    : 'max',
                                                                            'inNums_whm_min'    : 'min',
                                                                            'inNums_whm_mean'   : 'mean',
                                                                            'inNums_whm_std'    : 'std'
                                                                            })
    dt = dt.merge(tmp, on=['stationID','week','hour','minute'], how='left')

    tmp = dt.groupby(['stationID','week','hour','minute'], as_index=False)['outNums'].agg({
                                                                            'outNums_whm_max'    : 'max',
                                                                            'outNums_whm_min'    : 'min',
                                                                            'outNums_whm_mean'   : 'mean',
                                                                            'outNums_whm_std'    : 'std'
                                                                            })
    dt = dt.merge(tmp, on=['stationID','week','hour','minute'], how='left')

    tmp = dt.groupby(['stationID','week','hour'], as_index=False)['inNums'].agg({
                                                                            'inNums_wh_max'    : 'max',
                                                                            'inNums_wh_min'    : 'min',
                                                                            'inNums_wh_mean'   : 'mean',
                                                                            'inNums_wh_std'    : 'std'
                                                                            })
    dt = dt.merge(tmp, on=['stationID','week','hour'], how='left')

    tmp = dt.groupby(['stationID','week','hour'], as_index=False)['outNums'].agg({
                                                                            'outNums_wh_max'    : 'max',
                                                                            'outNums_wh_min'    : 'min',
                                                                            'outNums_wh_mean'   : 'mean',
                                                                            'outNums_wh_std'    : 'std'
                                                                            })
    dt = dt.merge(tmp, on=['stationID','week','hour'], how='left')
    return dt
	###################----------------------------------------------
	
total_data = totaldata()

freqcaller = findfreqcaller(total_data,50)

test = pd.read_csv('../input/Metro_testA/testA_submit_2019-01-29.csv')
test_28 = pd.read_csv('../input/Metro_testA/testA_record_2019-01-28.csv')
#创建常客身份标识
freqcalldict = dict(zip(freqcaller,[1]*len(freqcaller)))

data = test_28
data['freqcaller'] = data['userID'].map(freqcalldict)
data = coverttime(data)
for i in range(2,26):
    if i<10:
        s = str(0)+str(i)
    else:
        s = str(i)    
    #print(f'data {i},date 2019-01-{s}')   
    traindata_batch = pd.read_csv('../input/Metro_train/record_2019-01-'+s+'.csv')
    traindata_batch['freqcaller'] = traindata_batch['userID'].map(freqcalldict)
    traindata_batch = coverttime(traindata_batch)
    data = pd.concat([data,traindata_batch])
####-------------------------------------
#test = pd.merge(test,test_28[['stationID','lineID']].drop_duplicates(),on='stationID',how='left')
def fix_day(d):
    if d in [1,2,3,4]:
        return d
    elif d in [7,8,9,10,11]:
        return d - 2
    elif d in [14,15,16,17,18]:
        return d - 4
    elif d in [21,22,23,24,25]:
        return d - 6
    elif d in [28]:
        return d - 8
    
def get_refer_day(d):
    if d == 20:
        return 29
    else:
        return d + 1    
def recover_day(d):
    if d in [1,2,3,4]:
        return d
    elif d in [5,6,7,8,9]:
        return d + 2
    elif d in [10,11,12,13,14]:
        return d + 4
    elif d in [15,16,17,18,19]:
        return d + 6
    elif d == 20:
        return d + 8
    else:
        return d    
    
    
    # 剔除周末,并修改为连续时间
data = data[(data.day!=5)&(data.day!=6)]
data = data[(data.day!=12)&(data.day!=13)]
data = data[(data.day!=19)&(data.day!=20)]
data = data[(data.day!=26)&(data.day!=27)]

data['day'] = data['day'].apply(fix_day)

traindata = data[data['day']!=20]
#traindata = statis_feature(traindata)

sliddata = slidwindows(traindata,5)

#构建28号验证  stationID	lineID	week	day	hour	minstep	inNums	outNums
validdata = data[data['day']==20]
validdata = validdata[['stationID','lineID','week','day','hour','minute','inNums','outNums']]
now_day = 20
for j in range(1,6):
    tmp = traindata[traindata['day']==(now_day-j)]
    cols = tmp.columns.drop(['stationID','lineID','hour','minute'])
    tmp.drop(['week','day'],axis=1,inplace=True)
  #  tmp = tmp[['stationID','lineID','hour','minute','inNums','outNums']]
    for f in cols:
        tmp.rename(columns={f: f+'_last_'+str(j)}, inplace=True)
    validdata = pd.merge(validdata,tmp,on=['stationID','lineID','hour','minute'],how='left')

data_A = sliddata[sliddata['lineID']=='A']
data_B = sliddata[sliddata['lineID']=='B']
data_C = sliddata[sliddata['lineID']=='C']
valid_A = validdata[validdata['lineID']=='A']
valid_B = validdata[validdata['lineID']=='B']
valid_C = validdata[validdata['lineID']=='C']
del data_A['lineID']
del data_B['lineID']
del data_C['lineID']
del valid_A['lineID']
del valid_B['lineID']
del valid_C['lineID']
gc.collect()

##----------------------------------
import lightgbm as lgb
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 63,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_seed':0,
    'bagging_freq': 1,
    'verbose': 1,
    'reg_alpha':1,
    'reg_lambda':2
}

target = ['inNums','outNums']
features = sliddata.columns.drop(target+['lineID'])
X_train = sliddata[features].values
X_valid = validdata[features].values
y_train = sliddata['outNums'].values
y_valid = validdata['outNums'].values



def lgbtrain(train,evals,total_data=None):
    gbm = lgb.train(params,
                    train,
                    num_boost_round=10000,
                    valid_sets=[train,evals],
                    valid_names=['train','valid'],
                    early_stopping_rounds=200,
                    verbose_eval=1000
                    )
    if total_data!=None:
        gbm = lgb.train(params,
                    total_data,
                    num_boost_round=gbm.best_iteration,
                    valid_sets=[total_data],
                    valid_names=['train'],
                    verbose_eval=1000,
                    )
    return gbm

lgb_train = lgb.Dataset(X_train, y_train)
lgb_evals = lgb.Dataset(X_valid, y_valid , reference=lgb_train)
model = lgbtrain(lgb_train,lgb_evals)
































	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
