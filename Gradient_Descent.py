#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as mp
import tensorflow as tf


# In[3]:


x = tf.Variable(4.0)
with tf.GradientTape() as tape:
    y = x ** 2


# In[4]:


dy_dx= tape.gradient(y,x)
dy_dx


# In[7]:


w = tf.Variable(tf.random.normal((4,2)))
b = tf.Variable(tf.ones(2, dtype= tf.float32))
x = tf.Variable([[10.,20.,30.,40.]], dtype = tf.float32)
w,b,x


# In[8]:


# in order to avoid calling GradientTape multple time use persistent=True
with tf.GradientTape(persistent=True) as tape:
    y = tf.matmul(x,w) + b
    loss = tf.reduce_mean(y**2)
    


# In[9]:


[dl_dw,dl_db] = tape.gradient(loss, [w,b])


# In[10]:


# when keras is used GradientTape automatically records in forward pass of NN
layer = tf.keras.layers.Dense(2, activation = 'relu')
x = tf.constant([[10.,20.,30.]])


# In[11]:


with tf.GradientTape() as tape:
    y = layer(x)
    loss = tf.reduce_sum(y**2)
grad = tape.gradient(loss, layer.trainable_variables)


# In[12]:


# GradientTape watches only trainable variables - when varibles are marked as Trainable = False
w = tf.Variable(tf.random.normal((4,2)), trainable = False)


# In[ ]:




