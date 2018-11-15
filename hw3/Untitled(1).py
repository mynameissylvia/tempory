#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten,Dropout
from keras.models import Sequential
import sys


# In[2]:


def read_and_order(open_csv):
    data = pd.read_csv(open_csv)
    c = list(data).copy()
    label = []
    shape = []
    for i in range(len(data)):
        label.append(int(data[c[0]][i]))
        feature = np.fromstring(data[c[1]][i], dtype=int, sep=' ')
        shape.append(np.reshape(feature,(48,48)))
    y = np.asarray(label)
    x = np.asarray(shape)
    return y,x


# In[3]:


#train = sys.argv[1]
#test = sys.argv[2]
#sample = sys.argv[3]
train = 'train.csv'
order = read_and_order(train)


# In[4]:


def extant(a,n,length = 40):
    m = []
    l = len(a)
    c = 1.0*l/n
    original_l = int((l-length)/2)
    size_l = int((n-length)/2)
    count = 0
    for j in a:
        m1 =[]
        for k in range(n-1):
            current_n = k*c
            position = int(current_n)
            adding = current_n-position
            element = j[position]+j[position+1]*adding
            m1.append(element)
        m1.append(j[-1])
        if len(a)-count >original_l and count >=original_l:
            if size_l >0:
                m.append(m1[size_l:-1*size_l])
            else:
                m.append(m1)
        count+=1
    return m
            


# In[5]:


y_train = order[0]
x_train = order[1]
x_train = x_train.reshape(x_train.shape[0],48,48,1).astype('float32')/255
y_train= np_utils.to_categorical(y_train)


# In[34]:


#change to 40*40 ,added
x_train_m = order[1].reshape(order[1].shape[0],48,48).astype('float32')/255
x_train_mirrow = []
for i in x_train_m:
    a = i.tolist()
    new_m_1 = []  #DOWN RIGHT
    new_m_2 = []  #DOWN left
    new_m_3 = []  #UP RIGHT
    new_m_4 = []  #UP LEFT
    new_m_5 = []  #MIDDLE
    
    
    #REVERSE
    new_m_11 = [] 
    new_m_21 = []
    new_m_31 = []
    new_m_41 = []
    new_m_51 = []
    
    r = []
    count = 0
    for j in a:
        m = j.copy()
        n = j.copy()
        m.reverse()
        r.append(m)
        if count >=6:
            new_m_1.append(n[6:])
            new_m_11.append(m[6:])
            new_m_2.append(n[:-6])
            new_m_21.append(m[:-6])
        if len(a)-count >6:
            new_m_3.append(n[6:])
            new_m_31.append(m[6:])
            new_m_4.append(n[:-6])
            new_m_41.append(m[:-6])
        if len(a)-count >3 and count >=3:
            new_m_5.append(n[3:-3])
            new_m_51.append(m[3:-3])
        count+=1
    #add some extension
    x_train_mirrow.append(new_m_1)
    x_train_mirrow.append(new_m_2)
    x_train_mirrow.append(new_m_3)
    x_train_mirrow.append(new_m_4)
    x_train_mirrow.append(new_m_5)
    x_train_mirrow.append(new_m_11)
    x_train_mirrow.append(new_m_21)
    x_train_mirrow.append(new_m_31)
    x_train_mirrow.append(new_m_41)
    x_train_mirrow.append(new_m_51)
#    x_train_mirrow.append(extant(a,40))
#    x_train_mirrow.append(extant(r,40))
x_train_m = np.array(x_train_mirrow, dtype=np.float32)
x_train_m = x_train_m.reshape(x_train_m.shape[0],42,42,1).astype('float32')


# In[35]:


y_train = order[0]
y_train_m = [val for val in y_train.tolist() for _ in (0,1,2,3,4,5,6,7,8,9)]
y_train_m = np.array(y_train_m, dtype=np.float32)
y_train_m= np_utils.to_categorical(y_train_m)


# In[36]:


y_train_m.shape,x_train_m.shape


# In[ ]:



model = Sequential()
model.add(Conv2D(filters = 64,
                 kernel_size = (5,5),
                 padding = 'same',
                 input_shape =(42,42,1),
                 activation = 'relu'))
model.add(MaxPooling2D((4,4)))  
model.add(Conv2D(filters = 64,
                 kernel_size = (5,5),
                 padding = 'same',
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(7))
model.add(Activation('softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
training = model.fit(x_train_m, y_train_m,validation_split = 0.2, epochs=30, batch_size=128,verbose = 2)


# In[ ]:


test = 'test.csv'
testing = read_and_order(test)
x_test = testing[1].reshape(testing[1].shape[0],48,48).astype('float32')/255


# In[ ]:


x_test_m = []
for i in x_test:
    a = i.tolist()
    new_m_t = []
    count = 0
    for j in a:
        m = j.copy()
        if len(a)-count >3 and count >=3:
            new_m_t.append(m[3:-3])
        count+=1
    x_test_m.append(new_m_t)
x_test_m = np.array(x_test_m, dtype=np.float32)
x_test_m = x_test_m.reshape(x_test_m.shape[0],42,42,1).astype('float32')


# In[ ]:


prediction=model.predict_classes(x_test_m)


# In[ ]:


id_name = range(len(prediction))
outcome = {"id":id_name,
           "label":prediction}
out = pd.DataFrame(outcome)


# In[ ]:


sample = 'sample.csv'
out.to_csv(sample,index = False)


# In[ ]:


#0.654 filters = 20 kernel_size = (5,5) epochs = 30
#0.615 filters = 16 kernel_size = (5,5) epochs = 30
#0.667 filters = 32,kernel_size = (4,4),epochs = 30
#0.702 filters = 32,kernel_size = (5,5),epochs = 30
#0.649 filters = 32,kernel_size = (6,6),epochs = 20
#0.729 filters = 32,kernel_size = (6,6),epochs = 30
#0.739 filters = 48,kernel_size = (5,5),epochs = 30 batch_size=300
# filters = 64,kernel_size = (5,5),epochs = 30 batch_size=300


# In[53]:


model.save('current_best.h5')


# In[ ]:




