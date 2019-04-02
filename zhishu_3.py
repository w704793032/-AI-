import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt

creatVars = locals()

def coverttime(df):
    df['time'] = pd.to_datetime(df['time'],format='%Y-%m-%d %H:%M:%S')
    df['week'] = df['time'].dt.dayofweek
    timedelta = df['time'] - pd.datetime(df['time'].dt.year[0],df['time'].dt.month[0],df['time'].dt.day[0],0,0,0)
    df['minstep'] = timedelta.dt.seconds/600
    df['minstep'] = df['minstep'].astype(int)
    df['days_diff'] = df['time'] - pd.to_datetime('2019-01-01 00:00:00',format='%Y-%m-%d %H:%M:%S')
    df['days_diff'] = df['days_diff'].apply(lambda x:x.days)
    df['weekend'] = df['week'].apply(lambda x:1 if x>4 else 0 )
    return df
	
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

def change_rate(predata,lastdata):
    err = lastdata-predata
    rate = np.true_divide(err,lastdata)
    rate[np.isinf(rate)]=0
    rate[np.isnan(rate)]=0
    return rate

def indics_3(data,data_before,batch_size,alpha,beta,gama):

    s = np.array([0. for l in range(len(data)+batch_size)])
    t = np.array([0. for l in range(len(data)+batch_size)])
    p = np.array([0. for l in range(len(data)+batch_size)])
    mae_ave = []
    
    ##p为周期分量，每一个周期即144更新一次
    for i in range(0,len(data)-batch_size,batch_size):
        databatch = data[i:i+batch_size]
        realdatabatch = data[i+batch_size:i+batch_size*2]
        
        before_databatch = data_before[i:i+batch_size]
        before_next_databatch = data_before[i+batch_size:i+batch_size*2]
        #计算每一个时间段的变化率
        c_rate = change_rate(before_databatch,before_next_databatch)
        for j in range(batch_size):

            #初始值 
            ix = i+j
            if ix==0:
                s[ix] = databatch[j]
                t[ix] = databatch[j+1] - databatch[j]
            else:
                s[ix] = alpha*(databatch[j]/p[ix-1])+(1-alpha)*(s[ix-1]+t[ix-1])
                t[ix] = beta*(s[ix]-s[ix-1])+(1-beta)*t[ix-1]
            if ix<batch_size:
                p[ix] = 1
            else:
                p[ix] = gama*(databatch[j]/s[ix])+(1-gama)*p[ix-batch_size]
                
                
        #对数组负值进行清零
        s[ix+1-batch_size:ix+1][s[ix+1-batch_size:ix+1]<0]=0
        #通过变化率转化数组
        t_databatch = databatch + np.multiply(databatch,c_rate)
         #使用数据对预测值求均值
        #print(len((np.add(s[ix+1-batch_size:ix+1],databatch))/2.))
        s[ix+1-batch_size:ix+1] = (np.add(s[ix+1-batch_size:ix+1],t_databatch))/2.
        
        mae = MAE(s[ix+1-batch_size:ix+1],realdatabatch)
        if mae==0 or np.isnan(mae):
         #   print(f'batch {i}\'s result is error. mae:{mae}')
            continue
        #print(f'batch {i} mae:{mae}')
        mae_ave.append(mae)
    ##预测
    before_last_databatch = data_before[i+batch_size*2:i+batch_size*3]
    c_rate = change_rate(before_next_databatch,before_last_databatch)
    for j in range(batch_size):
        s[ix+j+1] = alpha*(realdatabatch[j]/p[ix+j])+(1-alpha)*(s[ix+j]+t[ix+j])
        t[ix+j+1] = beta*(s[ix+j+1]-s[ix+j])+(1-beta)*t[ix+j]
        p[ix+j+1] = gama*(realdatabatch[j]/s[ix+j+1])+(1-gama)*p[ix+j+1-batch_size]
    t_realdatabatch = realdatabatch + np.multiply(realdatabatch,c_rate)
    s[ix+1:ix+batch_size+1] = (np.add(s[ix+1:ix+batch_size+1],t_realdatabatch))/2.
    return s[ix+1:ix+batch_size+1],np.average(mae_ave)

#---------------------------------------------------------
test = pd.read_csv('../input/Metro_testA/testA_submit_2019-01-29.csv')
test['minstep'] = [j for i in range(81) for j in range(144)]

for sta in range(81):
    creatVars['data_in'+str(sta)] = []
    creatVars['data_out'+str(sta)] = []
for i in [8,15,22]:
    if i<10:
        s = str(0)+str(i)
    else:
        s = str(i)    
    print(f'traindata {i},date 2019-01-{s}')      
    traindata_batch = pd.read_csv('../input/Metro_train/record_2019-01-'+s+'.csv')
    traindata_batch = coverttime(traindata_batch) 
    singlesta = traindata_batch.groupby(['stationID','minstep'])['status'].agg(['count','sum']).reset_index()
    singlesta['outnums'] = singlesta['count']-singlesta['sum'] 
    singlesta['innums'] = singlesta['sum']
    dt_1 = pd.merge(test,singlesta,on=['stationID','minstep'],how='left')
    dt_1 = dt_1[['stationID','startTime','outnums','innums','minstep']]
    dt_1.fillna(0,inplace=True)
    for sta in range(81):
        creatVars['data_in'+str(sta)] = np.concatenate([creatVars['data_in'+str(sta)],dt_1[dt_1['stationID']==sta]['innums'].values])
        creatVars['data_out'+str(sta)] = np.concatenate([creatVars['data_out'+str(sta)],dt_1[dt_1['stationID']==sta]['outnums'].values])

##建立周期性头天的数据集，用来预测变化诚笃
for sta in range(81):
    creatVars['data_in_last'+str(sta)] = []
    creatVars['data_out_last'+str(sta)] = []
for i in [7,14,21,28]:
    if i<10:
        s = str(0)+str(i)
    else:
        s = str(i)    
    print(f'traindata {i},date 2019-01-{s}') 
    if i==28:
        traindata_batch = pd.read_csv('../input/Metro_testA/testA_record_2019-01-28.csv')
    else:
        traindata_batch = pd.read_csv('../input/Metro_train/record_2019-01-'+s+'.csv')
    traindata_batch = coverttime(traindata_batch) 
    singlesta = traindata_batch.groupby(['stationID','minstep'])['status'].agg(['count','sum']).reset_index()
    singlesta['outnums'] = singlesta['count']-singlesta['sum'] 
    singlesta['innums'] = singlesta['sum']
    dt_1 = pd.merge(test,singlesta,on=['stationID','minstep'],how='left')
    dt_1 = dt_1[['stationID','startTime','outnums','innums','minstep']]
    dt_1.fillna(0,inplace=True)
    for sta in range(81):
        creatVars['data_in_last'+str(sta)] = np.concatenate([creatVars['data_in_last'+str(sta)],dt_1[dt_1['stationID']==sta]['innums'].values])
        creatVars['data_out_last'+str(sta)] = np.concatenate([creatVars['data_out_last'+str(sta)],dt_1[dt_1['stationID']==sta]['outnums'].values])
#-------------------------------------------------------------
batch_size = 144
alpha = 0.4
beta = 0.05
gama = 0.05
s_in = []
s_out = []
s_in_ave_score = []
s_out_ave_score = []
for sta in range(81):
   # print(f'station {sta} inres')
    res_in,in_score = indics_3(data=creatVars['data_in'+str(sta)],data_before=creatVars['data_in_last'+str(sta)],batch_size=batch_size,alpha=alpha,beta=beta,gama=gama)
   # print(f'station {sta} outres')
    res_out,out_score = indics_3(data=creatVars['data_out'+str(sta)],data_before=creatVars['data_out_last'+str(sta)],batch_size=batch_size,alpha=alpha,beta=beta,gama=gama)

    if in_score==0 or np.isnan(in_score):
        print(f'station {sta}\'s in result is error. in_score:{in_score}')
    else:
        s_in_ave_score.append(in_score)
    if out_score==0 or np.isnan(out_score):
        print(f'station {sta}\'s out result is error. out_score:{out_score}')
    else:
        s_out_ave_score.append(out_score)

    if sta==0:
        s_in = res_in
        s_out = res_out
    else:
        s_in = np.concatenate((s_in,res_in))
        s_out = np.concatenate((s_out,res_out))
s_in_ave_score = np.average(s_in_ave_score)
s_out_ave_score = np.average(s_out_ave_score)
print(f'in restult score is {s_in_ave_score}, out result score is {s_out_ave_score}, average score is {(s_in_ave_score+s_out_ave_score)/2.}')

s_in[s_in<0]=0
s_out[s_out<0]=0
test['inNums'] = s_in
test['outNums'] = s_out

del test['minstep']
test.fillna(0,inplace=True)
test['inNums'] = test['inNums'].astype(int)
test['outNums'] = test['outNums'].astype(int)
test.to_csv('../output/res_indics_move_clear_zero.csv',index=False)

























































