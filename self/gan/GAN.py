'''
1.做参数传进generator函数的noise_img格式对不到（noise_img格式要是什么？除了使用占位符还能怎么解决？）
    格式：n*input_dim； 可以用格式适合的变量，但是placeholder更好训练
    1.之前无法显示noise值是因为tensorflow先定义流程，再在sess中运行，之前没有sess.run(tf.global_ariable_initializer())
2.generator与discrimination怎么合成一个网络（tf1.4里是否有Sequential？将两个网络连在一起是否有其他方法？）
    有，在tf.keras.models.Sequence
    如果没有，可以分别计算G(x)、D(x)、D(G(x))然后在Optimizer中设置var_list
'''
'''
运行效果：随机生成batch__size张数字图片
运行环境：tf_1.4.0
'''
import tensorflow as tf
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/data')  #导入mnist数据集

save_size = 10
learning_rate = 0.001
batch_size = 65
epochs = 301
input_dim = 100
output_shape = [28, 28]
model_path = './self/gan/model/generator.ckpt'

#生成器模型函数
#没用batch_normalization，要太多参数而且还没用
def get_generator(noise_img, img_shape=output_shape, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        # # noise = np.random.normal(0,1,input_dim)
        # # noise_1 = tf.Variable(tf.random_normal([input_dim]))
        # # noise_img = tf.layers.flatten(noise_img)

        # dense_1 = tf.layers.dense(noise_img, 256)
        # '''写法错误：'''
        # # dense_1 = tf.nn.leaky_relu(dense_1, alpha=0.2)        (错)

        # dense_1 = tf.maximum(0.01 * dense_1, dense_1)          # (对)
        # # mean, var = tf.nn.moments(dense_1, axes=[0])
        # # size = 3
        # # scale = tf.Variable(tf.ones([size]))
        # # shift = tf.Variable(tf.zeros([size]))
        # # epsilon = 0.001
        # # dense_1 = tf.nn.batch_normalization(dense_1, mean, var, shift, scale, epsilon)

        # dense_1 = tf.layers.dense(dense_1, units=512)
        # dense_1 = tf.maximum(0.01 * dense_1, dense_1)
        # # dense_1 = tf.nn.batch_normalization(dense_1, mean=0.8)

        # dense_1 = tf.layers.dense(dense_1, units=1024)
        # dense_1 = tf.maximum(0.01 * dense_1, dense_1)
        # # dense_1 = tf.nn.batch_normalization(dense_1, mean=0.8)

        # dense_1 = tf.layers.dense(dense_1, units=np.prod(img_shape))
        # dense_1 = tf.nn.tanh(dense_1)
        # # dense_1 = tf.reshape(dense_1, img_shape)

        # return dense_1
        #hidden layer
        hidden_1 = tf.layers.dense(noise_img, 512)
        #leaky RELU（relu的变形体）
        hidden_1 = tf.maximum(0.01 * hidden_1, hidden_1)
        #drop out
        hidden_1 = tf.layers.dropout(hidden_1, rate=0.2)

        #logits & outputs
        logits = tf.layers.dense(hidden_1, 784)
        outputs = tf.tanh(logits)

        # return logits, outputs
        return outputs

def get_discrimination(input_img, reuse=False):
    with tf.variable_scope('discrimination', reuse=reuse):
        # dense_1 = tf.layers.flatten(input_img)
        # input_img = tf.reshape(input_img, [None, 28*28])
        dense_1 = tf.layers.dense(input_img, 512)
        dense_1 = tf.maximum(0.01 * dense_1, dense_1)
        dense_1 = tf.layers.dense(dense_1, 256)
        dense_1 = tf.maximum(0.01 * dense_1, dense_1)
        dense_1 = tf.layers.dense(dense_1, 1)
        return dense_1
        # #隐藏层
        # hidden_1 = tf.layers.dense(input_img, 784)
        # hidden_1 = tf.maximum(0.01 * hidden_1, hidden_1)
        # #logits outputs
        # logits = tf.layers.dense(hidden_1, 1)
        # outputs = tf.sigmoid(logits)

        # return logits, outputs

def show_img(img):
    #(784) => (28,28)
    # one_image = img.reshape(28, 28)
    one_image = tf.reshape(img, shape=(28, 28))
    
    plt.axis('off')
    plt.imshow(one_image.eval(),cmap=cm.binary) #
    # fg_1 = plt.figure(1)
    plt.show()
    # plt.pause(1)
    # plt.close(fg_1)

def show_img(img):
    img = tf.reshape(img, shape=(-1,28,28))
    width = int(math.sqrt(int(img.shape[0]))+0.99)
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

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mode = 1        #0:train    1:test
if mode == 0:
    noise_holder = tf.placeholder(tf.float32, [None, input_dim])        #有效
    true_holder = tf.placeholder(tf.float32, [None, 784])
    # def get_inputs(real_size, noise_size):
    #     real_size = tf.placeholder(tf.float32, [None, real_size])       #图片大小
    #     noise_size = tf.placeholder(tf.float32, [None, noise_size])     #噪声大小
    #     return real_size, noise_size
    # noise_holder, true_holder = get_inputs(input_dim, 784)
    # true_label = np.ones([batch_size, 1])       #(错)
    # fake_label = np.zeros([batch_size, 1])
    # true_label = tf.ones_like(dis_result_true)      #(对)
    # fake_label = tf.zeros_like(dis_result_fake)
    gen_img = get_generator(noise_holder, output_shape) #gen_logits, 
    # print("gen_img.shape:", gen_img.get_shape())
    dis_result_true = get_discrimination(true_holder)   #, dis_re_true
    dis_result_fake = get_discrimination(gen_img, reuse=True)  #, dis_re_fake

    d_loss_true = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_result_true, labels=tf.ones_like(dis_result_true)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_result_fake, labels=tf.zeros_like(dis_result_fake)))
    d_loss = tf.add(d_loss_true, d_loss_fake)
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_result_fake, labels=tf.ones_like(dis_result_fake)))

    g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discrimination')
    # train_vars = tf.trainable_variables()

    # #generator生成器的神经网络
    # g_vars = [var for var in train_vars if var.name.startswith("generator")]
    # #discriminator判别器的神经网络
    # d_vars = [var for var in train_vars if var.name.startswith("discrimination")]


    d_train = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
    g_train = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            run_times = mnist.train.num_examples // batch_size
            for i in range(run_times):
                # batch = mnist.train.next_batch(batch_size)
                # x_batch = batch[0].reshape((batch_size, 28*28))
                # x_batch = x_batch * 2 - 1
                # noise_batch = np.random.uniform(-1, 1, size=(batch_size, input_dim))
                batch = mnist.train.next_batch(batch_size)  #一批数据（包括训练、验证图片、标签）
                x_batch = batch[0].reshape((batch_size, 784))  #训练图片数据
                #对图像像素进行scale，这是因为tanh输出的结果是介于(-1,1),real和fake图片共享discriminator的参数
                x_batch = x_batch*2 -1
                
                #随机初始化输入噪声
                noise_batch = np.random.uniform(-1,1,size=(batch_size, input_dim))

                # print(noise_batch[0],'\timage_shape:',x_batch[0])
                _ = sess.run(d_train, feed_dict={true_holder:x_batch, noise_holder:noise_batch})
                _ = sess.run(g_train, feed_dict={noise_holder:noise_batch})
            d_loss_train = sess.run(d_loss, feed_dict={noise_holder:noise_batch, true_holder:x_batch})
            g_loss_train = sess.run(g_loss, feed_dict={noise_holder:noise_batch})
            print("第", epoch, "轮epoch完毕\ng_loss:",g_loss_train, "\td_loss:", d_loss_train)

            # #验证generator
            # # def show_img(img):
            # #     #(784) => (28,28)
            # #     # one_image = img.reshape(28, 28)
            # #     one_image = tf.reshape(img, shape=(28, 28))
                
            # #     plt.axis('off')
            # #     plt.imshow(one_image.eval(),cmap=cm.binary)
            # #     plt.show()
            # noise_test_1 = np.random.uniform(-1,1,size=(batch_size, input_dim))
            # gen_img_2 = sess.run(get_generator(noise_holder, output_shape, reuse=True), feed_dict={noise_holder:noise_test_1})
            # show_img(gen_img_2[0])


            # print(gen_img_2[0])
            if epoch % save_size == 0 and epoch != 0:
                saver = tf.train.Saver(var_list=g_vars)
                saver.save(sess, model_path)
elif mode == 1:
    print("test\n")
    noise_test = np.random.uniform(-1, 1, size=(batch_size, input_dim))
    noise_img = tf.placeholder(tf.float32, [None, input_dim])
    # gen_img = get_generator(noise_holder_2, output_shape, reuse=tf.AUTO_REUSE) #gen_logits, 
    # tf.reset_default_graph()
    gen_img_1 = get_generator(noise_img, output_shape, reuse=tf.AUTO_REUSE) #gen_logits, 
    g_vars_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    saver_2 = tf.train.Saver(var_list=g_vars_2)     #
    # sess_2 = tf.Session()
    with tf.Session() as sess_2:
        tf.get_variable_scope().reuse_variables()
        saver_2.restore(sess_2, model_path)
        gen_img_1 = sess_2.run(get_generator(noise_img, output_shape), feed_dict={noise_img:noise_test})   
        show_img(gen_img_1)
elif mode == 2:
    mnist = input_data.read_data_sets('./data')
    show_mnist_img = mnist.train.next_batch(batch_size)
    show_img(show_mnist_img)
elif mode == 3:
    noise_test_3 = np.random.uniform(-1, 1, size=(16, input_dim))
    sess_3 = tf.Session()
    saver = tf.train.import_meta_graph(model_path)
    saver.restore(sess_3, tf.train.latest_checkpoint(model_path))
    graph = tf.get_default_graph()
    tensors = graph._get_tensor_by_tf_output()
    print('tensors:',tensors)
    # gen_img = graph.get_tensor_by_name('get_generator:0')
    # noise_img = graph.get_tensor_by_name('get_generator:0')   #noise_img
    # feed_dict = {noise_img:noise_test_3}
    # sess_3.run(gen_img, feed_dict=feed_dict)




# if __name__ == '__main__':

#     import os
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#     # #测试1.（格式不匹配）
#     # noise = np.random.normal(0,1,(input_dim))
#     # print('norse:', noise)
#     # output = get_generator(noise, output_shape)

#     # #测试2.(可以，但是好像要用placeholder)
#     # noise_1 = tf.Variable(tf.random_normal([2, input_dim], stddev=1, seed=1))
#     # sess = tf.Session()
#     # sess.run(tf.global_variables_initializer())
#     # # sess.run(noise_1)
#     # # print('noise_1:', sess.run(noise_1))
#     # output = get_generator(noise_1, output_shape)
#     # # output = tf.layers.dense(noise_1, units=128)
#     # # sess.run(output)

#     # #测试3.(最好的方法)
#     # noise = tf.placeholder(tf.float32, [None, input_dim])
#     # output = get_generator(noise, output_shape)
#     # print(output.get_shape())
#     train()
