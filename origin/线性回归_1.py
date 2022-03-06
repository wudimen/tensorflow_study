# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#随机生成1000个点
num_points = 1000
vectors_set = []
for i in range(num_points):
    x1 = np.random.normal(0.0,0.55)
    y1 = x1*0.1+0.3+np.random.normal(0.0,0.03)
    vectors_set.append([x1,y1])
    
#生成一些样本
x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

plt.scatter(x_data,y_data,c='r')
plt.show()


#生成以为的w矩阵，取值是【-1，1】之间的随机数
W = tf.Variable(tf.random_uniform([1],-1.0,1.0),name='W')

#生成以为的b矩阵，初始值是0
b=tf.Variable(tf.zeros([1]),name='b')

#经过计算得预算值y
y=W*x_data + b


#一预估值y和实际值y_data之间的均方差作为误差
loss = tf.reduce_mean(tf.square(y-y_data),name='loss')
#采用梯度下降法优化参数
optimizer = tf.train.GradientDescentOptimizer(0.5)#梯度下降步长
#训练的过程就是最小化这个误差值的过程
train = optimizer.minimize(loss,name='train')


sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

#初始化的W和b、loss是多少
print("W=",sess.run(W),"b = ",sess.run(b),"loss = ",sess.run(loss))

#执行20 次训练
for step in range(20):
    sess.run(train)
    #输出训练好的W与b
    print("W = ",sess.run(W),"b = ",sess.run(b),"loss = ",sess.run(loss))
    
#writer = tf.train.SummaryWriter("./tmp", sess.graph)



plt.scatter(x_data,y_data,c='r')
plt.plot(x_data,sess.run(W)*x_data+sess.run(b))
plt.show()












