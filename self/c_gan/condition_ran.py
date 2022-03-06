'''
运行环境：TF_1.4.1
效果：输入数字，输出图片（未训练完成，训练速度太慢）
'''
from math import sqrt
from re import A
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tqdm
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./data', one_hot=True)

epochs = 300
batch_size = 64
learning_rate = 0.001
noise_dim = 100
label_dim = 10
save_size = 1
output_shape = (28, 28)
model_path = './self/c_gan/model/generator.ckpt'

is_training = tf.placeholder(tf.bool, name='is_training')

def show_img(img):
    img = tf.reshape(img, shape=(-1,28,28))
    width = int(sqrt(int(img.shape[0]))+0.99)
    print('width:', width)
    plt.figure()
    plt.axis('off')
    # f, axarr = plt.subplots(width, width)
    # for j in range(width):
    #     for i in range(width):
    #         axarr[j+1][1+i].imshow(img[j][i].eval(), cmap=cm.binary)     #    #.eval():将tensor张量转换为数字常量
        
    for i in range(batch_size):
        plt.subplot(width,width,i+1)
        plt.imshow(img[i].eval(), cmap=cm.binary)
        plt.xticks([])
        plt.yticks([])
    plt.show()

#100+10=110 ==> 3*3*512 ==> 3,3,512 ==> 6,6,256 ==> 12,12,128 ==> 24,24,56 ==> 28,28,1
'''将噪声与标签合成一个变量， 然后一起经过神经网络，输出结果'''
def get_generator(noise, img_label, output_shape=output_shape, reuse=False, is_training=is_training):
    momentum = 0.9
    with tf.variable_scope('generator', reuse=reuse):
        noise_label = tf.concat([noise, img_label], axis=1, name='noise_label')
        dense_1 = tf.layers.dense(noise_label, units=3*3*512, name='dense_1')
        dense_reshape_1 = tf.reshape(dense_1, shape=(-1, 3, 3, 512), name='dense_reshape_1')
        dense_act_1 = tf.nn.relu(tf.contrib.layers.batch_norm(dense_reshape_1, is_training=is_training, decay=momentum))

        conv_1 = tf.layers.conv2d_transpose(dense_act_1, kernel_size=5, filters=256, strides=2, padding='same')
        conv_act_1 = tf.nn.relu(tf.contrib.layers.batch_norm(conv_1, is_training=is_training, decay=momentum))

        conv_2 = tf.layers.conv2d_transpose(conv_act_1, kernel_size=5, filters=128, strides=2, padding='same')
        conv_act_2 = tf.nn.relu(tf.contrib.layers.batch_norm(conv_2, is_training=is_training, decay=momentum))

        conv_3 = tf.layers.conv2d_transpose(conv_act_2, kernel_size=5, filters=56, strides=2, padding='same')
        conv_act_3 = tf.nn.relu(tf.contrib.layers.batch_norm(conv_3, is_training=is_training, decay=momentum))

        conv_4 = tf.layers.conv2d_transpose(conv_act_3, kernel_size=5, filters=1, strides=1, padding='valid', activation=tf.nn.tanh, name='g_result')

        # print('shape_1:', conv_4.get_shape())
        return conv_4

#[28,28,1]+[28,28,10] ==> [28,28,11]  ==> [14,14,64] ==> [7,7,128] ==> [4,4,256] ==> [2,2,512] ==> 2*2*512 ==> 1
'''将图片与label合成一个变量，然后经过神经网络，输出结果（'''
def get_discriminator(img, label, reuse=False, is_training=is_training):
    momentum = 0.9
    with tf.variable_scope('discriminator', reuse=reuse):
        dense_1 = tf.concat([img, label], axis=3)
        conv_1 = tf.layers.conv2d(dense_1, kernel_size=5, strides=2, filters=64, padding='same')
        conv_act_1 = tf.maximum(conv_1, conv_1*0.2)

        conv_2 = tf.layers.conv2d(conv_act_1, kernel_size=5, strides=2, filters=128, padding='same')
        conv_2 = tf.contrib.layers.batch_norm(conv_2 ,is_training=is_training, decay=momentum)
        conv_act_2 = tf.maximum(conv_2, conv_2*0.2)

        conv_3 = tf.layers.conv2d(conv_act_2, kernel_size=5, strides=2, filters=256, padding='same')
        conv_3 = tf.contrib.layers.batch_norm(conv_3, is_training=is_training, decay=momentum)
        conv_act_3 = tf.maximum(conv_3, conv_3*0.2)

        conv_4 = tf.layers.conv2d(conv_act_3, kernel_size=5, strides=2, filters=512, padding='same')
        conv_4 = tf.contrib.layers.batch_norm(conv_4, is_training=is_training, decay=momentum)
        conv_act_4 = tf.maximum(conv_4, conv_4*0.2)

        flat = tf.layers.flatten(conv_act_4)

        dense_2 = tf.layers.dense(flat, units=1)
        # print(dense_1.get_shape(), '\n', conv_1.get_shape(), '\n', conv_2.get_shape(), '\n', 
        #     conv_3.get_shape(), '\n', conv_4.get_shape(), '\n', flat.get_shape(), '\n', dense_2.get_shape())
        return dense_2

def to_onehot(label, num_classes = 10):
    one_hot = np.zeros([batch_size, num_classes])
    for i in range(batch_size):
        one_hot[i][label[i][0]] = 1
    return one_hot





noise_holder = tf.placeholder(tf.float32, shape=[None, noise_dim], name='noise_holder')
label_holder = tf.placeholder(tf.float32, shape=[None, label_dim], name='label_holder')
y_label_holder = tf.placeholder(tf.float32, [None, 28, 28, 10], name='y_label_holder')
img_holder = tf.placeholder(tf.float32, [None, 28, 28, 1], name='img_holder')

mode = 1
if mode == 0:
    x = get_generator(noise_holder, label_holder)
    y_real = get_discriminator(img_holder, y_label_holder)
    y_fake = get_discriminator(x, y_label_holder, reuse=True)

    g_vars = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
    d_vars = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(y_real), logits=y_real))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(y_fake), logits=y_fake))
    d_loss = 0.5*tf.add(d_loss_real, d_loss_fake)
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(y_fake), logits=y_fake))

    d_train = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
    g_train = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        is_training_real = True
        for epoch in range(epochs):
            batch_times = mnist.train.num_examples // batch_size
            for batch in tqdm.tqdm(range(batch_times)):

                noise = np.random.uniform(-1, 1, size=(batch_size, noise_dim))
                label = np.random.uniform(-1, 1, size=(batch_size, label_dim))
                img = np.random.uniform(-1, 1, size=(batch_size, 28, 28, 1))
                y_label = np.random.uniform(-1, 1, size=(batch_size, 28, 28, 10))

                img_train, label_train = mnist.train.next_batch(batch_size)
                img_train = np.reshape(img_train, [batch_size, 28, 28, 1])
                img_train = (img_train - 0.5) * 2
                yl = np.reshape(label_train, [batch_size, 1, 1, label_dim])
                yl = yl * np.ones([batch_size, 28, 28, label_dim])

                g_loss_result = sess.run(g_train, feed_dict={noise_holder:noise, label_holder:label_train, y_label_holder:yl, is_training:is_training_real})
                d_loss_result = sess.run(d_train, feed_dict={img_holder:img_train, label_holder:label_train, noise_holder:noise, noise_holder:noise, y_label_holder:yl, is_training:is_training_real})
            print("第", epoch, "轮epoch完毕\ng_loss:",g_loss_result, "\td_loss:", d_loss_result)

            if epoch % save_size == 0 and epoch != 0:
                saver = tf.train.Saver(var_list=g_vars)
                saver.save(sess, model_path)
                # gen_img = sess.run(x, feed_dict={noise_holder:noise, label_holder:label, is_training:is_training_real})
                # show_img(gen_img)
                dis_result = sess.run(y_real, feed_dict={img_holder:img, y_label_holder:y_label, is_training:is_training_real})
elif mode == 1:
    print('test\n')
    is_training_real = False
    noise_test = np.random.uniform(-1, 1, size=(batch_size, noise_dim))
    label_test = np.random.randint(1,2, size=(batch_size, 1))
    label_onehot_test = to_onehot(label_test)

    gen_img = get_generator(noise_holder, label_holder, reuse=tf.AUTO_REUSE) #tf.AUTO_REUSE
    g_vars = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
    saver = tf.train.Saver(var_list=g_vars)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.get_variable_scope().reuse_variables()     #
        saver.restore(sess, model_path)
        gen_img_result = sess.run(get_generator(noise_holder, label_holder), feed_dict={noise_holder:noise_test, label_holder:label_onehot_test, is_training:is_training_real})
        show_img(gen_img_result)

elif mode == 3:
    label = np.random.randint(0, 10, size=(batch_size, 1))
    label_onehot = to_onehot(label)
    print(label_onehot.shape)
    for i in range(batch_size):
        print(i, ':\t', label[i], ':\t', label_onehot[i], '\n')