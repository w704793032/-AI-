import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

from keras import backend as k
from keras.layers import *
from keras import Sequential

creatVars = locals()

def coverttime(df):
    df['time'] = pd.to_datetime(df['time'],format='%Y-%m-%d %H:%M:%S')
    df['week'] = df['time'].dt.dayofweek
    timedelta = df['time'] - pd.datetime(df['time'].dt.year[0],df['time'].dt.month[0],df['time'].dt.day[0],0,0,0)
    df['minstep'] = timedelta.dt.seconds/600
    df['minstep'] = df['minstep'].astype(int)
    return df
	
	
	
def buildlstmmodel():
    model = Sequential()
    model.add(LSTM(input_dim=151, output_dim=200, return_sequences=True))
    #print(model.layers)
    model.add(LSTM(300, return_sequences=False))
    model.add(Dense(output_dim=144))
    model.add(Activation('sigmoid'))
    model.compile(loss='mse', optimizer='rmsprop')
    return model
	
def get_traindata(df,empty_file):
    singlesta = df.groupby(['stationID','minstep'])['status'].agg(['count','sum']).reset_index()
    singlesta['outnums'] = singlesta['count']-singlesta['sum'] 
    singlesta['innums'] = singlesta['sum']
    empty_file['minstep'] = [j for i in range(81) for j in range(144)]
    singlesta = pd.merge(empty_file,singlesta,on=['stationID','minstep'],how='left')
    singlesta.fillna(0, inplace=True)
    #df['days_diff'] = df['time'] - pd.to_datetime('2019-01-01 00:00:00',format='%Y-%m-%d %H:%M:%S')
    singlesta['week'] = df['week'].values[:singlesta.shape[0]]
    #singlesta['days_diff'] = df['days_diff'].values[:singlesta.shape[0]]
    #singlesta['days_diff'] = singlesta['days_diff'].apply(lambda x:x.days)
    
    tdata = singlesta['innums'].values.reshape(1,-1)
    #days_diff = singlesta['days_diff'].unique()[0]
    week = singlesta['week'].unique()[0]
    #tdata = np.append(tdata,days_diff)
    tdata = np.append(tdata,week)
    
    return tdata
	
test = pd.read_csv('../input/Metro_testA/testA_submit_2019-01-29.csv')

	
for i in [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]:
    if i<10:
        s = str(0)+str(i)
    else:
        s = str(i)    
    print(f'traindata {i},date 2019-01-{s}')      
    traindata_batch = pd.read_csv('../input/Metro_train/record_2019-01-'+s+'.csv')
    traindata_batch = coverttime(traindata_batch)
    if i ==2:
        predata = get_traindata(traindata_batch,test)
    else:
        predata = np.row_stack((predata,get_traindata(traindata_batch,test)))
    del traindata_batch
    gc.collect()
	
	
scaler = MinMaxScaler()
#对流量数据归一化
predata[:,:-1] = scaler.fit_transform(predata[:,:-1])

weekenc = OneHotEncoder(sparse=False)
#对星期onehot
predata = np.hstack((predata[:,:-1],weekenc.fit_transform(predata[:,-1].reshape(-1,1))))
	

	
sequence_length = 8 #窗口大小 以前7天的数据预测后一天的数据  7+1
data = []
for i in range(predata.shape[0]-sequence_length+1):
    data.append(predata[i:i+sequence_length,:])

reshapedata = np.array(data)

x = reshapedata[:,:-1]
y = reshapedata[:,-1,:11664]

split_boundary = int(reshapedata.shape[0] * 0.8)
# 构建训练集
train_x = x[: split_boundary]
# 构建测试集(原数据的后20%)
test_x = x[split_boundary:]
# 训练集标签
train_y = y[: split_boundary]
# 测试集标签
test_y = y[split_boundary:]


#对每个station单独构建模型并训练预测
for i in range(81):
    print(f'++++++++++++++++batch {i}++++++++++++++++++++')
    train_x_batch = np.concatenate((train_x[:,:,i*144:(i+1)*144],train_x[:,:,-7:]),axis=2)
    train_y_batch = train_y[:,i*144:(i+1)*144]
    test_x_batch = np.concatenate((test_x[:,:,i*144:(i+1)*144],test_x[:,:,-7:]),axis=2)
    test_y_batch = test_y[:,i*144:(i+1)*144]
    creatVars['model_'+str(i)] = buildlstmmodel()
    creatVars['model_'+str(i)].fit(train_x_batch,train_y_batch,epochs=10,validation_split=0.1)
    loss = creatVars['model_'+str(i)].evaluate(test_x_batch,test_y_batch)
    print(f'batch {i} loss: {loss}')

##predict
##先预测26号，然后预测27号数据，最后聚合28号数据预测29号数据
usedata = predata[-7:,:]
for i in range(81):
    usedata_batch = np.concatenate((usedata[:,i*144:(i+1)*144],usedata[:,-7:]),axis=1)
    usedata_batch = usedata_batch.reshape(1,usedata_batch.shape[0],usedata_batch.shape[1])
    if i==0:
        result = creatVars['model_'+str(i)].predict(usedata_batch)
    else:
        result = np.concatenate((result,creatVars['model_'+str(i)].predict(usedata_batch)),axis=1)


##添加星期特征
result = np.concatenate((result,weekenc.transform(5)),axis=1)
usedata = np.concatenate((usedata,result),axis=0)

usedata = usedata[-7:,:]
for i in range(81):
    usedata_batch = np.concatenate((usedata[:,i*144:(i+1)*144],usedata[:,-7:]),axis=1)
    usedata_batch = usedata_batch.reshape(1,usedata_batch.shape[0],usedata_batch.shape[1])
    print('batch',i)
    if i==0:
        result = creatVars['model_'+str(i)].predict(usedata_batch)
    else:
        result = np.concatenate((result,creatVars['model_'+str(i)].predict(usedata_batch)),axis=1)


result = np.concatenate((result,weekenc.transform(6)),axis=1)
usedata = np.concatenate((usedata,result),axis=0)

##增加28日数据
traindata_batch_28 = pd.read_csv('../input/Metro_testA/testA_record_2019-01-28.csv')
traindata_batch_28 = coverttime(traindata_batch_28)
predata28 = get_traindata(traindata_batch_28,test)
predata28 = predata28.reshape(1,-1)

predata28[:,:-1] = scaler.fit_transform(predata28[:,:-1])
#对星期onehot
predata28 = np.hstack((predata28[:,:-1],weekenc.transform(predata28[:,-1].reshape(-1,1))))

usedata = np.concatenate((usedata,predata28),axis=0)

usedata = usedata[-7:,:]
for i in range(81):
    usedata_batch = np.concatenate((usedata[:,i*144:(i+1)*144],usedata[:,-7:]),axis=1)
    usedata_batch = usedata_batch.reshape(1,usedata_batch.shape[0],usedata_batch.shape[1])
    print('batch',i)
    if i==0:
        result = creatVars['model_'+str(i)].predict(usedata_batch)
    else:
        result = np.concatenate((result,creatVars['model_'+str(i)].predict(usedata_batch)),axis=1)

# 对标准化处理后的数据还原
result = scaler.inverse_transform(result)
testfile = pd.read_csv('../input/Metro_testA/testA_submit_2019-01-29.csv')
testfile['inNums'] = result[0]
testfile['inNums'] = testfile['inNums'].astype(int)

















