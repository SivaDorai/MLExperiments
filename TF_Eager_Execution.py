#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
tf.executing_eagerly()


# In[2]:


a = tf.constant([[1,2],[3,4]])
a


# In[3]:


b = tf.add(a,2)
b


# In[4]:


x= tf.Variable([100,100], name="x")
x


# In[5]:


y = a*x+b
y


# In[6]:


# to run tensorboard
get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir="./logs" --port 6060')


# In[ ]:




