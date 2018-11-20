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
from PIL import Image
from keras.preprocessing import image
import os


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
x_train = x_train.reshape(x_train.shape[0],48,48,1).astype('uint8')/255
y_train= np_utils.to_categorical(y_train)


# In[6]:


datagen = image.ImageDataGenerator(featurewise_center=True,
                                   featurewise_std_normalization=True,
                                   rotation_range=30,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range = 0.3,)


# In[42]:


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
    new_m_6 = []
    new_m_7 = []
    new_m_8 = []
    new_m_9 = []
    
    #REVERSE
    new_m_11 = [] 
    new_m_21 = []
    new_m_31 = []
    new_m_41 = []
    new_m_51 = []
    new_m_61 = []
    new_m_71 = []
    new_m_81 = []
    new_m_91 = []
    
    
    r = []
    count = 0
    for j in a:
        m = j.copy()
        n = j.copy()
        m.reverse()
        r.append(m)
        if count >=2 and len(a)-count >6:
            new_m_6.append(n[2:-6])
            new_m_61.append(m[2:-6])
            new_m_7.append(n[6:-2])
            new_m_71.append(m[6:-2])
        if count >=6 and len(a)-count >2:
            new_m_8.append(n[2:-6])
            new_m_81.append(m[2:-6])
            new_m_9.append(n[6:-2])
            new_m_91.append(m[6:-2])
        if count >=8:
            new_m_1.append(n[8:])
            new_m_11.append(m[8:])
            new_m_2.append(n[:-8])
            new_m_21.append(m[:-8])
        if len(a)-count >8:
            new_m_3.append(n[8:])
            new_m_31.append(m[8:])
            new_m_4.append(n[:-8])
            new_m_41.append(m[:-8])
        if len(a)-count >4 and count >=4:
            new_m_5.append(n[4:-4])
            new_m_51.append(m[4:-4])
        count+=1
    #add some extension
    x_train_mirrow.append(new_m_1)
    x_train_mirrow.append(new_m_2)
    x_train_mirrow.append(new_m_3)
    x_train_mirrow.append(new_m_4)
    x_train_mirrow.append(new_m_5)
    x_train_mirrow.append(new_m_6)
    x_train_mirrow.append(new_m_7)
    x_train_mirrow.append(new_m_8)
    x_train_mirrow.append(new_m_9)
    x_train_mirrow.append(new_m_11)
    x_train_mirrow.append(new_m_21)
    x_train_mirrow.append(new_m_31)
    x_train_mirrow.append(new_m_41)
    x_train_mirrow.append(new_m_51)
    x_train_mirrow.append(new_m_61)
    x_train_mirrow.append(new_m_71)
    x_train_mirrow.append(new_m_81)
    x_train_mirrow.append(new_m_91)
    
print(len(x_train_mirrow))


# In[43]:


x_train_m = np.array(x_train_mirrow, dtype=np.float32)
x_train_m = x_train_m.reshape(x_train_m.shape[0],40,40,1).astype('float32')


# In[33]:


y_train = order[0]
y_train_m = [val for val in y_train.tolist() for _ in (0,1,2,3,4,5,6,7,8,9)]
y_train_m = np.array(y_train_m, dtype=np.float32)
y_train_m= np_utils.to_categorical(y_train_m)


# In[34]:


y_train_m.shape,x_train_m.shape


# In[37]:


datagen = image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range = 0.3,)
epochs =30
batch_size = 64


# In[38]:



model = Sequential()
model.add(Conv2D(filters = 64,
                 kernel_size = (4,4),
                 padding = 'same',
                 input_shape =(40,40,1),
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


# In[39]:


#from keras.utils import generic_utils


# Data Augmentation后，数据变多了，因此我们需要更的训练次数
#for e in range(epochs*4):
 #   print('Epoch', e)
  #  print('Training...')
   # progbar = generic_utils.Progbar(x_train_m.shape[0])
    #batches = 0
    
    #for x_batch, y_batch in datagen.flow(x_train_m, y_train_m, batch_size=batch_size, shuffle=True):
     #   loss,train_acc = model.train_on_batch(x_batch, y_batch)
      #  batches += x_batch.shape[0]
       # if batches > x_train_m.shape[0]:
        #    break
        #progbar.add(x_batch.shape[0], values=[('train loss', loss),('train acc', train_acc)])


# In[20]:


test = 'test.csv'
testing = read_and_order(test)
x_test = testing[1].reshape(testing[1].shape[0],48,48).astype('float32')/255


# In[ ]:





# In[21]:


x_test_m = []
for i in x_test:
    a = i.tolist()
    new_m_t = []
    count = 0
    for j in a:
        m = j.copy()
        if len(a)-count >4 and count >=4:
            new_m_t.append(m[4:-4])
        count+=1
    x_test_m.append(new_m_t)
x_test_m = np.array(x_test_m, dtype=np.float32)
x_test_m = x_test_m.reshape(x_test_m.shape[0],40,40,1).astype('float32')


# In[22]:


prediction=model.predict_classes(x_test_m)


# In[23]:


id_name = range(len(prediction))
outcome = {"id":id_name,
           "label":prediction}
out = pd.DataFrame(outcome)


# In[24]:


sample = 'sample_2.csv'
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




