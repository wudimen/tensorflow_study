# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:37:21 2022

@author: LJ
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('data/', one_hot=True)  #下载的手写数字数据集



#NETWORK TOPOLOGIES
n_hidden_1 = 256  #隐藏层1层数
n_hidden_2 = 128  #隐藏层2层数
n_input    = 784  #输入数据大小（28*28）
n_classes  = 10   #需输出的数据大小（属于各个数的概率值/数的种类）


#INPUTS AND OUTPUTS
x = tf.placeholder("float", [None,n_input])  #从mnist中取出一些数据（输入数据）
y = tf.placeholder("float", [None,n_classes])  #从mnist中取出一些数据（输出数据）

#NETWORK PARAMETERS
stddev = 0.1                         #定义参数（权重与偏移量）
weights = {
        'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1],stddev=stddev)),
        'w2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2],stddev=stddev)),
        'out': tf.Variable(tf.random_normal([n_hidden_2,n_classes],stddev=stddev))
        }
biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
        }
print("NETWORK READY")



def multilayer_perceptron(_X, _weight, _biases):  #处理输入数据的函数
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weight['w1']),_biases['b1']))  #a
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weight['w2']),_biases['b2']))
    return (tf.matmul(layer_2,_weight['out']) + _biases['out'])

#PREDICTION
pred = multilayer_perceptron(x,weights,biases)   #预测值（用输入数据通过现存参数计算得到的预测值）

#LOSS AND OPTIMIZER
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))  #损失值（实际值y与预测值pred通过softmax计算得到的值）
optm = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(cost)   #利用梯度下降法对参数进行调整
corr = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))            #argmax：通过分析各个树的概率值判断出是哪个数，看预测值与实际值是否相同
accr = tf.reduce_mean(tf.cast(corr,"float"))       #计算上面比较结果的均值，看准确性
        
#INITIALIZER
init = tf.global_variables_initializer()
print("FUNCTION READY")


training_epochs = 20  #迭代20次，每次都含有mnist里所有的数据
batch_size = 100   #批处理数量，每次计算100个数据，可减轻污点影响，防止过拟合
display_step = 4  #每隔4个epoch展览一次数据
#LAUNCH THE GRAPH
sess = tf.Session()
sess.run(init)
#OPTIMIZE  使最优化
for epoch in range(training_epochs):  #循环20次（epoch数量）
    avg_cost = 0  #平均损失值，每次循环结束后的
    total_batch = int(mnist.train.num_examples/batch_size)   #总共要执行多少次批处理（总数量/每次计算的数量）
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)  #每次从mnist里提取批处理数量个数据，，，如果是自己的数据，要自己定义这个函数
        feeds = {x: batch_xs, y: batch_ys}  #字典结构，内部有x与y，用读取的数据进行填充
        sess.run(optm, feed_dict = feeds)  #optm就是利用梯度下降法调整参数的过程“函数”，每调用一次就相当于训练了一次
        avg_cost += sess.run(cost, feed_dict=feeds)  #计算当前损失值总和，用于计算平均损失值
    avg_cost = avg_cost / total_batch
    #DISPLAY
    if(epoch+1) % display_step == 0:
        print("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))  #打印信息
        feeds = {x: batch_xs, y: batch_ys }  #用train的数据进行计算准确度
        train_acc = sess.run(accr, feed_dict=feeds)  #计算训练值的平均准确度
        print("TRAIN ACCURACY: %.3f"% (train_acc))
        feeds = {x: mnist.test.images, y: mnist.test.labels}  #用test的数据进行计算准确度
        test_acc = sess.run(accr, feed_dict=feeds)  #计算测试值的平均准确度
        print("TEST ACCURACY: %.3f" % (test_acc))
        
saver = tf.train.Saver()
save_path = saver.save(sess,"d:/sess/mnist_cnn_test_1/")
print("sess had saved in ",save_path)
print("OPTIMIZATION FINISHED")



'''
# NETWORK TOPOLOGIES
n_hidden_1 = 256 
n_hidden_2 = 128 
n_input    = 784 
n_classes  = 10  

# INPUTS AND OUTPUTS
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
    

# NETWORK PARAMETERS
stddev = 0.1
weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=stddev)),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=stddev))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
print ("NETWORK READY")


def multilayer_perceptron(_X, _weights, _biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['w1']), _biases['b1'])) 
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights['w2']), _biases['b2']))
    return (tf.matmul(layer_2, _weights['out']) + _biases['out'])

# PREDICTION
pred = multilayer_perceptron(x, weights, biases)

# LOSS AND OPTIMIZER
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) 
optm = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost) 
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))    
accr = tf.reduce_mean(tf.cast(corr, "float"))

# INITIALIZER
init = tf.global_variables_initializer()
print ("FUNCTIONS READY")


training_epochs = 20
batch_size      = 100
display_step    = 4
# LAUNCH THE GRAPH
sess = tf.Session()
sess.run(init)
# OPTIMIZE
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)
    # ITERATION
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feeds = {x: batch_xs, y: batch_ys}
        sess.run(optm, feed_dict=feeds)
        avg_cost += sess.run(cost, feed_dict=feeds)
    avg_cost = avg_cost / total_batch
    # DISPLAY
    if (epoch+1) % display_step == 0:
        print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        feeds = {x: batch_xs, y: batch_ys}
        train_acc = sess.run(accr, feed_dict=feeds)
        print ("TRAIN ACCURACY: %.3f" % (train_acc))
        feeds = {x: mnist.test.images, y: mnist.test.labels}
        test_acc = sess.run(accr, feed_dict=feeds)
        print ("TEST ACCURACY: %.3f" % (test_acc))
print ("OPTIMIZATION FINISHED")


'''