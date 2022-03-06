'''
定义模型：68-73
计算损失值：75-87
定义优化器：91-101
运行优化器（训练模型）：134-136
'''
from asyncio import sleep
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/data')  #导入mnist数据集

#通过输入要生成图片、初始化噪声的大小生成生成图片与噪声的占位符
def get_inputs(real_size, noise_size):
    real_size = tf.placeholder(tf.float32, [None, real_size])       #图片大小
    noise_size = tf.placeholder(tf.float32, [None, noise_size])     #噪声大小
    return real_size, noise_size

#生成 生成器（俩全连接层）（最后输出out_dim形状的数据与激活后的数据）
def get_generator(noise_img, n_units, out_dim, reuse=False, alpha=0.01):
    with tf.variable_scope("generator", reuse=reuse):       #用于定义创建变量（层）的操作的上下文管理器（防止上下文有相同的变量、若上文有相同变量则自动改名）
        #hidden layer
        hidden_1 = tf.layers.dense(noise_img, n_units)
        #leaky RELU（relu的变形体）
        hidden_1 = tf.maximum(alpha * hidden_1, hidden_1)
        #drop out
        hidden_1 = tf.layers.dropout(hidden_1, rate=0.2)

        #logits & outputs
        logits = tf.layers.dense(hidden_1, out_dim)
        outputs = tf.tanh(logits)

        return logits, outputs

#生成 判别器（俩全连接层）
def get_discriminator(img, n_units, reuse=False, alpha=0.01):
    with tf.variable_scope("discriminator", reuse=reuse):
        #隐藏层
        hidden_1 = tf.layers.dense(img, n_units)
        hidden_1 = tf.maximum(alpha * hidden_1, hidden_1)
        #logits outputs
        logits = tf.layers.dense(hidden_1, 1)
        outputs = tf.sigmoid(logits)

        return logits, outputs


img_size = mnist.train.images[0].shape[0]   #mnist数据集形状
# print(img_size) #784

noise_size = 100    #初始化噪声形状

g_units = 128   #

d_units = 128   #

learning_rate = 0.001   #学习率

alpha = 0.01    #


tf.reset_default_graph()

real_img, noise_img = get_inputs(img_size, noise_size)  #生成占位符

#generator
g_logits, g_outputs = get_generator(noise_img, g_units, img_size)   #g_logits:无激活函数的生成器    g_outputs:有激活函数的生成器

#discriminator
d_logits_real, d_outputs_real = get_discriminator(real_img, d_units)    #通过正式数据生成的无/有激活函数的判别器
d_logits_fake, d_output_fake = get_discriminator(g_outputs, d_units, reuse=True)    ##通过生成器生成的数据生成的无/有激活函数的判别器

#计算discriminator的loss
#识别真实图片的损失值（实际值与标签值的差值==>转换为概率==>求平均值）
#tf.ones_like/zeros_like：生成与参数形状一样的数字全是1/0的变量：用来生成标签值
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,labels=tf.ones_like(d_logits_real)))

#识别生成图片的损失值
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,labels=tf.zeros_like(d_logits_fake)))

#总体损失值
d_loss = tf.add(d_loss_real, d_loss_fake)

#计算generator的损失值
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_logits_fake)))



#优化器
train_vars = tf.trainable_variables()

#generator生成器的神经网络
g_vars = [var for var in train_vars if var.name.startswith("generator")]
#discriminator判别器的神经网络
d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

#optimizer生成器与判别器的优化器
d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)


'''
变量准备完毕，开始训练
'''

#训练
batch_size = 64
epochs = 300
n_sample = 25

#用于存储测试样例
samples = []
samples = []
#储存loss
losses = []
losses = []
#用于保存生成器模型
saver = tf.train.Saver(var_list=g_vars)
#开始训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        for batch in range(mnist.train.num_examples//batch_size):   #每个epoch循环次数
            batch = mnist.train.next_batch(batch_size)  #一批数据（包括训练、验证图片、标签）
            batch_images = batch[0].reshape((batch_size, 784))  #训练图片数据
            #对图像像素进行scale，这是因为tanh输出的结果是介于(-1,1),real和fake图片共享discriminator的参数
            batch_images = batch_images*2 -1
            
            #随机初始化输入噪声
            batch_noise = np.random.uniform(-1,1,size=(batch_size, noise_size))
            # print('*********************************', batch_noise[0],'\timage_shape:',batch_images[0], '##############################################')
            # sleep(1000000)
            # break
            #运行优化器
            _ = sess.run(d_train_opt, feed_dict={real_img:batch_images, noise_img:batch_noise})
            _ = sess.run(g_train_opt, feed_dict={noise_img:batch_noise})

        # noise_test_1 = np.random.uniform(-1,1,size=(batch_size, noise_size))
        # gen_img_2 = sess.run(get_generator(noise_img, g_units, img_size, reuse=True), feed_dict={noise_img:noise_test_1})
        # # show_img(gen_img)
        # print(gen_img_2[0])
        # sample_noise = np.random.uniform(-1, 1, size=(n_sample, noise_size))
        # gen_samples = sess.run(get_generator(noise_img, g_units, img_size, reuse=True),feed_dict={noise_img:sample_noise})

        #每一轮结束计算损失值值
        train_loss_d = sess.run(d_loss, feed_dict={real_img:batch_images,noise_img:batch_noise})
        #真实图片损失值
        train_loss_d_real = sess.run(d_loss_real, feed_dict={real_img:batch_images,noise_img:batch_noise})
        #生成图片损失值
        train_loss_d_fake = sess.run(d_loss_fake, feed_dict={real_img:batch_images,noise_img:batch_noise})
        #generator loss
        train_loss_g = sess.run(g_loss, feed_dict={noise_img:batch_noise})

        print("Epoch {}/{}...".format(epoch+1, epochs),
        "判别器损失：{:.4f}(判别真实的：{:.4f} + 判别生成的：{:.4f})...".format(train_loss_d, train_loss_d_real, train_loss_d_fake),
        "生成器损失：{:.4f}".format(train_loss_g))
        
        #保存信息
        losses.append((train_loss_d, train_loss_d_real, train_loss_d_fake, train_loss_g))

        #保存样本
        sample_noise = np.random.uniform(-1, 1, size=(n_sample, noise_size))
        gen_samples = sess.run(get_generator(noise_img, g_units, img_size, reuse=True),feed_dict={noise_img:sample_noise})
        samples.append(gen_samples)

        saver.save(sess, './chechpoints/generator.ckpt')


#保存到本地
with open('train_samples.pkl', 'wb') as f:
    pickle.dump(samples, f)