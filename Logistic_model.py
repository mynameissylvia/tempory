
# coding: utf-8

# In[39]:

import pandas as pd
import numpy as np
import csv
import math
import sys


# In[45]:

def change_feature(x):
    x = x.drop(['LIMIT_BAL'],axis = 1)
    for i in range(6):
        x[str(i+1)] = np.where(x['EDUCATION']==i+1, 1, 0)
    x['Marry'] = np.where(x['MARRIAGE']==2, 1, 0)
    x['Single'] = np.where(x['MARRIAGE']==1, 1, 0)
    x['Unknown'] = np.where(x['MARRIAGE']==3, 1, 0)
    x = x.drop(['EDUCATION','SEX','MARRIAGE'],axis = 1)
    return x


# In[50]:

def training(inp_csvx,inp_csvy):
    x_train = pd.read_csv(inp_csvx)
    y_train = pd.read_csv(inp_csvy)
    x_train = change_feature(x_train)
    print(x_train)
    X = x_train.values
    Y = y_train.T.values[0].astype(float)
    
    minx = np.min(X, axis=0)
    maxx = np.max(X, axis=0)
    X = (X - minx) / (maxx - minx)
    
    w =  np.zeros(len(X[0]))
    m = X.shape[0]
    b = 0
    lamba = 10^-6
    
    lr = (2.79)*(10^-5)
    lr_b = 0
    lr_w = 0
    current_F1 = 0
    run = 10200
    costs = []
    count = 0
    regul_p = 10^-9
    for i in range(run):
        
        tem_y = np.dot(X,w)+b
        h0 = 1.0/(1 + np.exp(-1.0 * tem_y))
        dw = (np.dot(X.T, np.transpose(h0 - Y)) + regul_p * w) / m  
        db = np.mean((h0 - Y))
        lr_w += dw ** 2
        lr_b += db ** 2
        w = w - lr * dw / (np.sqrt(lr_w)+lamba)
        b = b - lr * db / (np.sqrt(lr_b)+lamba)

    return w,b,minx,maxx


# In[51]:

def test(test_csv,output_csv,a):
    w=a[0]
    b=a[1]
    minx = a[2]
    maxx = a[3]
    test = pd.read_csv(test_csv)
    test = change_feature(test)
    Xt = test.values
    Xt = (Xt - minx) / (maxx - minx)
    Yt = np.dot(Xt,w)+b
    h0t = 1.0/(1 + np.exp(-1.0 * Yt))

    Prediction = [1 if i >= 0.5 else 0 for i in h0t]
    id_ = np.array([])
    for i in range(len(Yt)):
        id_=np.append(id_,'id_'+str(i))
    final = pd.DataFrame(np.concatenate(([id_],[Prediction])).transpose(),columns = ['id','Value'])
    final.to_csv(output_csv,index = False)
    return Prediction


# In[52]:

train_x = sys.arg[1]
train_y = sys.arg[2]
test_x = sys.arg[3]
out = sys.arg[4]

a = training(train_x,train_y)
t=test(test_x,out,a)

