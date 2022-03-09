#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
tf.debugging.set_log_device_placement(True)


# In[2]:


tf.executing_eagerly()


# In[3]:


x0= tf.constant(3)
print(x0)


# In[5]:


x0.dtype


# In[6]:


x0.shape


# In[7]:


#provides numpy equivalent of the tensor
x0.numpy()


# In[8]:


# 1 dim array - Vector
x1= tf.constant([1.1,2.2,3.3,4.4])
x1


# In[10]:


# 2 dim array - matrix
x2= tf.constant([[1,2,3,4],[5,6,7,8]])
x2


# In[12]:


# change dtype
x2 = tf.cast(x2, tf.float32)
x2


# In[13]:


x3 = np.array([[10,20],[20,30],[40,50]])
x3


# In[14]:


#change numpy array to tensor
x3 = tf.convert_to_tensor(x3)
x3


# In[15]:


# all numpy operations can be performed on tensors  like sqrt, square


# In[17]:


# initializes tensor with zeros for the shape and dtype specified
# similary tf.ones can be used to fill with ones.
t0 = tf.zeros([2,3],tf.int32)
t0


# In[18]:


v1 = tf.Variable([[1,2,3],[4,5,6]])
v1


# In[22]:


#dtype specified explicitly
v2 = tf.Variable ([[1,2,3],[4,5,6]], dtype=tf.float32)
v2
v3 = tf.Variable ([[11,12,13],[14,15,16]], dtype=tf.float32)


# In[24]:


# performs addition of varibles and the output is a tensor. Note v2 and v3 are both float 
#- The add throws error when we add int and float
tf.add(v3,v2) 


# In[25]:


# to convert variable to tensor
tf.convert_to_tensor(v1)


# In[26]:


# assign method can be used to update the varible entirely 
v1.assign([[2,5,7],[92,1,56]])
v1


# In[27]:


v1[0,0].assign(33) # assigns specific cell
v1


# In[28]:


# assign_add and assign_sub can be used as well


# In[ ]:





# In[ ]:




