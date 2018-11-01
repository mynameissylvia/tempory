
# coding: utf-8

# In[107]:

import pandas as pd
import numpy as np
import csv
import math
import sys


# In[108]:

def change_feature(x):
    x = x.drop(['LIMIT_BAL'],axis = 1)
    for i in range(6):
        x[str(i+1)] = np.where(x['EDUCATION']==i+1, 1.0, 0.0)
    x['Marry'] = np.where(x['MARRIAGE']==2, 1.0, 0.0)
    x['Single'] = np.where(x['MARRIAGE']==1, 1.0, 0.0)
    x['Unknown'] = np.where(x['MARRIAGE']==3, 1.0, 0.0)
    x = x.drop(['EDUCATION','SEX','MARRIAGE'],axis = 1)
    return x


# In[180]:

def training(inp_csvx,inp_csvy):
    
    x_train = pd.read_csv(inp_csvx)
    y_train = pd.read_csv(inp_csvy)
    x_train = change_feature(x_train)
    x_t = x_train.copy()
    maxx = x_t.max(axis = 0)
    minx = x_t.min(axis = 0)
    x_t = (x_t-minx)/(maxx-minx)
    
    y_1 = y_train.index[y_train['Y']==1]
    y_2 = y_train.index[y_train['Y']==0]
    x_1 = x_t.loc[y_1]
    x_2 = x_t.loc[y_2]
    
    u1 = x_1.mean(axis = 0).values
    u2 = x_2.mean(axis = 0).values
    
    N1 = len(y_1.tolist())
    N2 = len(y_2.tolist())
    
    s1 = x_1.cov().values
    s2 = x_2.cov().values
    sigma = (s1*N1+s2*N2)/(N1+N2)
    
    w = np.dot((u1-u2).T,np.linalg.inv(sigma)).T
    b = -0.5*np.dot(np.dot(u1.T,np.linalg.inv(sigma)),u1)+0.5*np.dot(np.dot(u2.T,np.linalg.inv(sigma)),u2)+np.log(N1/N2)
    
    print(w)
    return w,b,maxx,minx


# In[181]:

def test(test_csv,output_csv,a):
    w=a[0]
    b=a[1]
    maxx = a[2]
    minx = a[3]
    test = pd.read_csv(test_csv)
    test = change_feature(test)
    
    test = (test-minx)/(maxx-minx)
    
    Xt = test.values
    Yt = np.dot(Xt,w)+b
    print(Yt)
    
    h0t = 1.0/(1 + np.exp(-1.0 * Yt))

    Prediction = [1 if i >= 0.5 else 0 for i in h0t]
    id_ = np.array([])
    for i in range(len(Yt)):
        id_=np.append(id_,'id_'+str(i))
    final = pd.DataFrame(np.concatenate(([id_],[Prediction])).transpose(),columns = ['id','Value'])
    final.to_csv(output_csv,index = False)
    return Prediction


# In[182]:

train_x = 'train_x.csv'
train_y = 'train_y.csv'
test_x = 'test_x.csv'
out = 'sample_submission_.csv'
#train_x = sys.arg[1]
#train_y = sys.arg[2]
#test_x = sys.arg[3]
#out = sys.arg[4]

a = training(train_x,train_y)
t=test(test_x,out,a)


# In[183]:

sum(t)


# In[ ]:



