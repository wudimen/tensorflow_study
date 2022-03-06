# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:13:23 2022

@author: LJ
"""
import pandas as pd
import numpy as np #df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD')) df[[1,2]] #KeyError: '[1 2] not in index' df.iloc[[1,2]] # A B C D #1 25 97 78 74 #2 6 84 16 21

#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf


#SETTINGS
LEARNING_RATE = 1e-4  #学习率、步长
#setto 20000 on local encironment to get 0.99 acurary
TRAINING_ITERATIONS = 2500  #测试集数据数量、训练迭代数量

DROPOUT = 0.5  #舍弃率（防止过拟合、加快训练速度）
BATCH_SIZE = 50  #单次训练数量                                       #总共训练50次，每次都包含全部的数据集

#set to 0 to train on all available data
VALIDATION_SIZE = 2000   #验证集大小

#image number to output
IMAGE_TO_DISPLAY = 10  #用于展示的图片



#read training data from CSV file
data = pd.read_csv('C:/Users/LJ/train.csv')  #导入训练数据

print('data({0[0]},{0[1]})'.format(data.shape))  #查看数据类型
print(data.head())  #查看数据内容




images = data.iloc[:,1:].values
images = images.astype(np.float)

#convert from [0,255] => [0.0,1.0]
images = np.multiply(images,1.0/255.0)

print('images({0[0]},{0[1]})'.format(images.shape))



image_size = images.shape[1]
print('image_size => {0}'.format(image_size))

#in this case all image are square
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

print('image_width => {0}\nimage_height => {1}'.format(image_width,image_height))




def display(img):
    #(784) => (28,28)
    one_image = img.reshape(image_width,image_height)
    
    plt.axis('off')
    plt.imshow(one_image,cmap=cm.binary)

print("********************************:\n", images[IMAGE_TO_DISPLAY], "\n**********************************")
#output image
display(images[IMAGE_TO_DISPLAY])

labels_flat = data.iloc[:,0].values.ravel()  #水平显示的标签

print('labels_flat({0})'.format(len(labels_flat))) #打印有多少个label
print('labels_flat[{0}] => {1}'.format(IMAGE_TO_DISPLAY,labels_flat[IMAGE_TO_DISPLAY]))  #打印要展览的image是数字几


labels_count = np.unique(labels_flat).shape[0]  #总共的数字数量（10）。unique：去重复数字

print('labels_count => {0}'.format(labels_count))  #打印有多数字



#convert class labels from scalars to one_hot vectors
def dense_to_one_hot(labels_dense,num_classes):  #将数字转换为“独热”类型数据（要转换的数字集，数字的类型数量）
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels)*num_classes
    labels_one_hot = np.zeros((num_labels,num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

labels = dense_to_one_hot(labels_flat,labels_count)
labels = labels.astype(np.uint8)

print('labels({0[0]}, {0[1]})'.format(labels.shape))
print('labels[{0}]  => {1}'.format(IMAGE_TO_DISPLAY,labels[IMAGE_TO_DISPLAY]))


#slpit data into training & validation
validation_images = images[:VALIDATION_SIZE]  #用于测试的测试集（2000个）
validation_labels = labels[:VALIDATION_SIZE]  #用于测试的测试集的标签

train_images = images[VALIDATION_SIZE:]  #用于训练的训练集（40000个）
train_labels = labels[VALIDATION_SIZE:]  #用于训练的训练集的标签

print(labels.shape)
print(validation_labels.shape)


def weight_variable(shape):  #初始化权重参数的函数
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):  #初始化偏移量的函数
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

    
    #x: 输入图像数据
    #W： filter
    #strides：tf在四维图像上计算
    #strides第一个参数表示在batchsize上的滑动，一般指定为1就可以，后边三个参数依次为宽、高、颜色通道上的滑动，
    #一般只修改中间两个值，也就是在宽高上的滑动
    #strides的第四个参数1表示通道上的滑动
def conv2d(x,W):  #用于卷积的函数
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')


# pooling
# [[0,3],
#  [4,2]] => 4

# [[0,1],
#  [1,1]] => 1
#tf.nn.max_pool(value,kszie,strides,padding,name=None)
#value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
#ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
#strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
#padding：和卷积类似，可以取'VALID' 或者'SAME'
def max_pool_2x2(x):  #用于池化的函数
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')



#input & output of NN
    
#images
    
x = tf.placeholder('float', shape=[None, image_size])  #单次训练的数据集
#labels
y_ = tf.placeholder('float', shape=[None, labels_count])  #单次训练的数据集的标签值

'''
总训练步骤：
卷积 =》 激活 =》 池化 =》 卷积2 =》 激活2 =》 池化2 =》 全连接（激活+全连接） =》 dropout =》 

'''

#第一次卷积激活池化
#卷积核初始化
W_convl = weight_variable([5,5,1,32])  #卷积核高、宽、通道数、卷积核数量
#偏置量初始化
b_convl = bias_variable([32])

#(40000,785) => (40000,28,28,1)
image = tf.reshape(x,[-1,image_width,image_height,1])  #将数据转换为可卷积的数据类型
print("image.shape=",image.shape)

h_convl = tf.nn.relu(conv2d(image, W_convl) + b_convl)  #卷积+激活
#print("h_convl.get_shape())# => (40000,14,14,32)
print("h_convl.shape=",h_convl.shape)
h_pool1 = max_pool_2x2(h_convl)  #池化
#print(h_pools.get_shape())# => (40000,14,14,32)
print("h_pool1.shape=", h_pool1.shape)



#第二次卷积激活池化
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)  #卷积+激活
print("h_conv2.shape=", h_conv2.shape)

h_pool2 = max_pool_2x2(h_conv2)  #池化
print("h_pool2.shape=",h_pool2.shape)




#全连接层
W_fc1 = weight_variable([7*7*64,1024])
print("W_fc1.shape=", W_fc1.shape)
b_fc1 = bias_variable([1024])

#(40000,7,7,64) => (40000,3136)
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])  #将池化结果转换为可进行全连接操作的单行数据集
print("h_pool2_flat.shape", h_pool2_flat.shape)
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)  #全连接
print("h_fc1.shape", h_fc1.shape)



#dropout
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#readout layer for deep net
W_fc2 = weight_variable([1024, labels_count])
b_fc2 = bias_variable([labels_count])

y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  #全连接2+softmax：将各结果转换为各数字的概率

#cost function
cross_entropy = -tf.reduce_sum(y_*tf.log(y))  #交叉熵

#optimisation function
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)  #梯度下降法调整参数，训练参数用的

#evaluation
corrent_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))  #当前预测结果

accuracy = tf.reduce_mean(tf.cast(corrent_prediction, 'float'))  #统计预测结果准确度



#prediction function
#[0.1,0.2,0.3,0.4,0.1,0.3,0.2,……] => 3
predict = tf.argmax(y,1)


epochs_completed = 0 #epoch完成数量
index_in_epoch = 0  #当前epoch数
num_examples = train_images.shape[0]  #训练集图片的总共数量

#server data by batches
def next_batch(batch_size):  #提取下一批训练集的函数
    global train_images    #global：对全局变量重新赋值，在函数内操作函数外的变量
    global train_labels
    global index_in_epoch
    global epochs_completed
    
    start = index_in_epoch
    index_in_epoch += batch_size
    
    #when all training data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        #finished epoch
        epochs_completed += 1
        #shufle the data
        perm = np.arange(num_examples)  #返回0，1，2，3，4，……，num_examples
        np.random.shuffle(perm)  #打乱perm
        train_images = train_images[perm]  #打乱训练集
        train_labels = train_labels[perm]
        #start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end],train_labels[start:end]


#start Tensorflow session 开始tensorflow session操作
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()

sess.run(init)


#visualisation variable  #用来看的数据
train_accuracies = []  #训练准确度
validation_accuracies = []  #测试准确度
x_range = []  

display_step = 1 #打印处理结果的参次（每display_step打印一次）

for i in range(TRAINING_ITERATIONS):  #执行2500次
    #get new epoch
    batch_xs, batch_ys = next_batch(BATCH_SIZE) #提取下一批的训练数据
    #chech progress on every 1st,2st……100st……step
    if i%display_step == 0 or (i+1) == TRAINING_ITERATIONS: #每display_step次或到了最后一次打印一下数据
        train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_:batch_ys,keep_prob:1.0})
        if(VALIDATION_SIZE):
            validation_accuracy = accuracy.eval(feed_dict ={x:validation_images[0:BATCH_SIZE],
                                                            y_:validation_labels[0:BATCH_SIZE],
                                                            keep_prob:1.0})
            print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d'%(train_accuracy, validation_accuracy, i))
            validation_accuracies.append(validation_accuracy)
            
        else:
            print('training_accuracy => %.4f for step %d'%(train_accuracy, i))
        train_accuracies.append(train_accuracy)
        x_range.append(i)
        
        #increase display_step
        if i%(display_step*10) == 0 and i:
            display_step *= 10
            
    #train on epoch上面一大坨都是用于打印信息的，下面这一句才是训练参数的
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})
        
#保存sess
saver = tf.train.Saver()
save_path = saver.save(sess,"d:/sess/mnist_cnn_test_2/")
print("sess had saved in ",save_path)     
        
        
#check final accuracy on validation set
if(VALIDATION_SIZE):
    validation_accuracy = accuracy.eval(feed_dict={x: validation_images,
                                                   y_: validation_labels,
                                                   keep_prob: 1.0})
    print('validation_acuracy => %.4f'%validation_accuracy)
    plt.plot(x_range, train_accuracies,'-b',label='Training')
    plt.plot(x_range, validation_accuracies,'-g', label='Validation')
    plt.legend(loc='lower right', frameon=False)
    plt.ylim(ymax=1.1, ymin=0.7)
    plt.ylabel('accuracy')
    plt.xlabel('step')
    plt.show()
    





