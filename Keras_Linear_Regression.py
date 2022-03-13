#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers


# In[2]:


w_true = 2
b_true  = 0.5
x = np.linspace(0,3, 130)
y = w_true * x + b_true + np.random.randn(*x.shape)*0.5


# In[3]:


x.shape, y.shape


# In[4]:


x = pd.DataFrame(x, columns=['x'])
y = pd.DataFrame(y, columns=['y'])
x.head()
y.head()


# In[5]:


model = keras.Sequential([layers.Dense(1, input_shape=(1,), activation='linear')])
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
model.compile(loss='mse', metrics=['mse'], optimizer= optimizer)


# In[6]:


model.fit(x,y,epochs=100)


# In[11]:


y_pred = model.predict(x)


# In[14]:


plt.figure(figsize=(10,8))
plt.scatter(x,y,c='blue',label='original')
plt.plot(x,y_pred,c='red',label='Fitted')
plt.legend()
plt.show()


# In[ ]:




