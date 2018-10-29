# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 15:23:16 2018

@author: freeze
"""
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.mixture import GaussianMixture
import warnings

warnings.filterwarnings('ignore')
color=sns.color_palette()
df=pd.read_csv('C:/Users/freeze/Desktop/OriginData/FormatData/final.csv',encoding='gbk')


def PlotAttribute(arr):
    """
    df['prf'].max=500000.0
    df['prf'].min=1.0
    存在nan的值，选择丢弃之后绘图
    """
    prf_values=np.sort((df[arr].dropna().values))
    ulimit=np.percentile(prf_values,85)
    #remove invalid data point
    #prf_clean_values=[x for x in prf_values if x<ulimit] 
    df_prf_clean_values=df[arr].dropna().ix[df[arr]<ulimit]
    plt.figure(figsize=(8,6))
    #plt.scatter(range(len(prf_clean_values)),prf_clean_values)
    plt.xlabel(arr,fontsize=12)
    #plt.show()
    sns.distplot(df_prf_clean_values.values,bins=50,kde=True)
    plt.show()

def grouplmplot(df):
    """
    填充nan
    """
    df.fillna(0.01)
    pr=df.groupby('mode')
    L=[]
    for a,b in pr:
        L.append(b.shape[0])
        try:
            sns.lmplot(x='prf',y='rf',data=b)
        except:
            print ('error occur')
        plt.xlabel(a,fontsize=12)
        plt.show()
    return len(pr),L


def TrainOneClassifier(df,classA,classB):
    #训练并返回一个效果最好的分类器
    Acontent=df[df['mode']==classA]
    Bcontent=df[df['mode']==classB]
    #先尝试两个属性
    Ax=Acontent[['prf','rf']]
    Bx=Bcontent[['prf','rf']]
    Ay=Acontent['mode']
    By=Bcontent['mode']
    x_train=pd.concat([Ax,Bx],axis=0,ignore_index=True)
    y_train=pd.concat([Ay,By],axis=0,ignore_index=True)
    clf=svm.SVC(cache_size=500)
    clf_param={'C':[1,2],'kernel':['linear','poly','rbf','sigmoid']}
    clf_grid=GridSearchCV(clf,clf_param)
    clf_grid.fit(x_train,y_train)
    
    print (clf_grid.best_params_)
    print (clf_grid.best_score_)
    #print (clf_grid.best_estimator_.predict(x_train))
    return clf_grid.best_estimator_

def low_bound(n,L):
    #计算下标
    NL=list()
    NL.append(L[0])
    for i in range(1,len(L)):
        NL.append(L[i]+L[i-1])
    c=0
    for arr in NL:
        if(arr<n):
            c=c+1
        else:
            break
    return c

def DataArrangement(origin_data,attr,n_samples=300):
    #参数说明，origin_data DataFrame原生对象 attr类名 n_samples默认300 生成个数
    DD=origin_data[origin_data['mode']==attr]
    data=DD[['prf','rf']]
    gmm=GaussianMixture(covariance_type='full')
    gmm_param={'n_components':[1,2,3,4,5]}
    gmm_grid=GridSearchCV(gmm,gmm_param)
    gmm_grid.fit(data)
    
    best_estimator=gmm_grid.best_estimator_
    weights=list(best_estimator.weights_)
    means=best_estimator.means_
    covas=best_estimator.covariances_
    print ('weights: '+str(weights))
    print ('means '+str(means))
    print ('covariances '+str(covas))
    #shape=len(weights)
    generator=list()
    for i in range(n_samples):
        ret=random.random()
        index=low_bound(ret,weights)
        generator.append(np.random.multivariate_normal(means[index],covas[index],1)[0])
    return np.array(generator)
"""
PlotAttribute('prf')
PlotAttribute('rf')
PlotAttribute('pw')
PlotAttribute('js')
PlotAttribute('B')
PlotAttribute('cm')
PlotAttribute('cl')
PlotAttribute('mf')
"""
#a,L=grouplmplot(df)

"""
xx=df[df['mode']=='radar_AN/APG-63   ']
yy=df[df['mode']=='radar_AN/APG-66   ']
plt.scatter(x=xx['prf'],y=xx['rf'],color='red')
plt.scatter(x=yy['prf'],y=yy['rf'],color='blue')

plt.show()
"""


"""
   数据扩增 data Argumation
"""
#1.针对属性prf rf 做一些众数填充
df.prf[df.prf.isnull()]=df.prf.dropna().mode().values
df.rf[df.rf.isnull()]=df.rf.dropna().mode().values

#2.填充之后 prf/rf都是 28201条属性,然后移除重复项 
df=df.drop_duplicates()
 
#3.factorize mode属性
df['mode']=pd.factorize(df['mode'])[0]


#4.Generator对某一类数据做增强
generator_data=DataArrangement(df,1)
print ('-------------------ok-------------------')
print ('')
print ('-------------------classifier-------------------')
#5.discramiter 判断是否继续算参数
classifier=TrainOneClassifier(df,0,1)

classifier.predict(generator_data)









