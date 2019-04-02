import matplotlib.pyplot as plt
def pltinoutsta(df,plot=True):
    #统计每一个十分钟时间段，进出站的人数
    insta=[]
    outsta=[]
    minsteps=df['minstep'].unique()
    for minstep in minsteps:
        insta.append((df[df['minstep']==minstep]['status']==0).sum())
        outsta.append((df[df['minstep']==minstep]['status']==1).sum())
    if plot==True:
        plt.plot(minstep,insta,color='red',label='insta')
        plt.plot(minstep,outsta,color='green',label='outsta')
        plt.show()
    else:
        pass
    return minsteps,insta,outsta
	
fig = plt.figure(figsize=(80,240))
for i in range(1,26):
    if i<10:
        s = str(0)+str(i)
    else:
        s = str(i)    
    print(f'traindata {i},date 2019-01-{s}')  
    traindata_batch = pd.read_csv('../input/Metro_train/record_2019-01-'+s+'.csv')
    traindata_batch = coverttime(traindata_batch)
    minsteps,instas,outstas = pltinoutsta(traindata_batch,plot=False)
    creatVars['ax'+s] = fig.add_subplot(9,3,i)
    creatVars['ax'+s].plot(minsteps,instas,color='red',label='insta')
    creatVars['ax'+s].plot(minsteps,outstas,color='green',label='outsta')

    creatVars['ax'+s].set_title(f'date 2019-01-{s}',fontsize=30)
    creatVars['ax'+s].legend(['insta','outsta'],fontsize=30)
plt.tick_params(labelsize=30)
plt.xlabel('phase',size=30)
plt.ylabel('value',size=30)
plt.savefig('../output/totalflow.png')
plt.show()