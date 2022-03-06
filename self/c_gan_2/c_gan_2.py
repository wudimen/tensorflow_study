'''
1.做参数传进generator函数的noise_img格式对不到（noise_img格式要是什么？除了使用占位符还能怎么解决？）
    格式：n*input_dim； 可以用格式适合的变量，但是placeholder更好训练
    1.之前无法显示noise值是因为tensorflow先定义流程，再在sess中运行，之前没有sess.run(tf.global_ariable_initializer())
2.generator与discrimination怎么合成一个网络（tf1.4里是否有Sequential？将两个网络连在一起是否有其他方法？）
    有，在tf.keras.models.Sequence
    如果没有，可以分别计算G(x)、D(x)、D(G(x))然后在Optimizer中设置var_list
'''
'''
运行效果：输入数字类型，生成batch_size张对应数字图片（在25行改数字类型）
训练了300左右轮epoch
运行环境：tf_1.4.0
'''
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import math

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/data', one_hot=True)  #导入mnist数据集

num = 8
save_size = 10
learning_rate = 0.001
batch_size = 64
epochs = 301
input_dim = 100
label_dim = 10
output_shape = [28, 28]
model_path = './self/c_gan_2/model/generator.ckpt'

#生成器模型函数
#没用batch_normalization，要太多参数而且还没用
def get_generator(noise_img, label, img_shape=output_shape, reuse=False):
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
        noise_img = tf.concat([noise_img, label], axis=1)
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

def get_discrimination(input_img, label, reuse=False):
    with tf.variable_scope('discrimination', reuse=reuse):
        # dense_1 = tf.layers.flatten(input_img)
        # input_img = tf.reshape(input_img, [None, 28*28])
        input_img = tf.concat([input_img, label], axis=1)
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

def show():
    with open('train_samples.pkl', 'rb') as f:
        samples = pickle.load(f)
    fig, axes = plt.subplots(figsize=(7, 7), nrows=5, ncols=5, sharey=True, sharex=True)
    for ax, image in zip(axes.flatten(), samples):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(image.reshape((28, 28)), cmap='Greys_r')
    plt.show()


def to_onehot(label, num_classes = 10):
    one_hot = np.zeros([batch_size, num_classes])
    for i in range(batch_size):
        one_hot[i][label[i][0]] = 1
    return one_hot

def pred_prev(num=0):
    noise_test = np.random.uniform(-1, 1, size=(batch_size, input_dim))
    label_test =  np.random.randint(0, 10, size=(batch_size, 1))    #np.ones([batch_size, 1])
    for x in range(batch_size):
        label_test[x][0] = num
    label_onehot_test = to_onehot(label_test)
    return noise_test, label_onehot_test

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mode = 1        #0:train    1:test  2:  3:  4:训练已训练模型
if mode == 0:
    noise_holder = tf.placeholder(tf.float32, [None, input_dim])        #有效
    true_holder = tf.placeholder(tf.float32, [None, 784])
    label_holder = tf.placeholder(tf.float32, [None, label_dim])
    # def get_inputs(real_size, noise_size):
    #     real_size = tf.placeholder(tf.float32, [None, real_size])       #图片大小
    #     noise_size = tf.placeholder(tf.float32, [None, noise_size])     #噪声大小
    #     return real_size, noise_size
    # noise_holder, true_holder = get_inputs(input_dim, 784)
    # true_label = np.ones([batch_size, 1])       #(错)
    # fake_label = np.zeros([batch_size, 1])
    # true_label = tf.ones_like(dis_result_true)      #(对)
    # fake_label = tf.zeros_like(dis_result_fake)
    gen_img = get_generator(noise_holder, label_holder, output_shape) #gen_logits, 
    # print("gen_img.shape:", gen_img.get_shape())
    dis_result_true = get_discrimination(true_holder, label_holder)   #, dis_re_true
    dis_result_fake = get_discrimination(gen_img, label_holder, reuse=True)  #, dis_re_fake

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
            for i in tqdm(range(run_times)):
                # batch = mnist.train.next_batch(batch_size)
                # x_batch = batch[0].reshape((batch_size, 28*28))
                # x_batch = x_batch * 2 - 1
                # noise_batch = np.random.uniform(-1, 1, size=(batch_size, input_dim))
                batch, label = mnist.train.next_batch(batch_size)  #一批数据（包括训练、验证图片、标签）
                x_batch = batch.reshape((batch_size, 784))  #训练图片数据
                #对图像像素进行scale，这是因为tanh输出的结果是介于(-1,1),real和fake图片共享discriminator的参数
                x_batch = x_batch*2 -1
                
                #随机初始化输入噪声
                noise_batch = np.random.uniform(-1,1,size=(batch_size, input_dim))

                # print(noise_batch[0],'\timage_shape:',x_batch[0])
                _ = sess.run(d_train, feed_dict={true_holder:x_batch, label_holder:label, noise_holder:noise_batch})
                _ = sess.run(g_train, feed_dict={label_holder:label, noise_holder:noise_batch})
            d_loss_train = sess.run(d_loss, feed_dict={noise_holder:noise_batch, label_holder:label, true_holder:x_batch})
            g_loss_train = sess.run(g_loss, feed_dict={label_holder:label, noise_holder:noise_batch})
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
    noise_test, label_onehot_test = pred_prev(num)
    noise_img = tf.placeholder(tf.float32, [None, input_dim])
    label_holder = tf.placeholder(tf.float32, [None, label_dim])
    # gen_img = get_generator(noise_holder_2, output_shape, reuse=tf.AUTO_REUSE) #gen_logits, 
    # tf.reset_default_graph()
    gen_img_1 = get_generator(noise_img, label_holder, output_shape, reuse=tf.AUTO_REUSE) #gen_logits, 
    g_vars_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    saver_2 = tf.train.Saver(var_list=g_vars_2)     #
    # sess_2 = tf.Session()
    with tf.Session() as sess_2:
        tf.get_variable_scope().reuse_variables()
        saver_2.restore(sess_2, model_path)
        gen_img_1 = sess_2.run(get_generator(noise_img, label_holder, output_shape), feed_dict={noise_img:noise_test, label_holder:label_onehot_test})   
        show_img(gen_img_1)
elif mode == 2:
    mnist = input_data.read_data_sets('./data')
    show_mnist_img = mnist.train.images[1000]
    print(show_mnist_img)
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
elif mode == 4:
    noise_holder = tf.placeholder(tf.float32, [None, input_dim])        #有效
    true_holder = tf.placeholder(tf.float32, [None, 784])
    label_holder = tf.placeholder(tf.float32, [None, label_dim])
    gen_img = get_generator(noise_holder, label_holder, output_shape) #gen_logits, 
    # print("gen_img.shape:", gen_img.get_shape())
    dis_result_true = get_discrimination(true_holder, label_holder)   #, dis_re_true
    dis_result_fake = get_discrimination(gen_img, label_holder, reuse=True)  #, dis_re_fake

    d_loss_true = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_result_true, labels=tf.ones_like(dis_result_true)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_result_fake, labels=tf.zeros_like(dis_result_fake)))
    d_loss = tf.add(d_loss_true, d_loss_fake)
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_result_fake, labels=tf.ones_like(dis_result_fake)))

    g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discrimination')

    d_train = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
    g_train = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

    saver = tf.train.Saver(var_list=g_vars)     #
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path)

        batch, label = mnist.train.next_batch(batch_size)  #一批数据（包括训练、验证图片、标签）
        x_batch = batch.reshape((batch_size, 784))  #训练图片数据
        x_batch = x_batch*2 -1
        noise_batch = np.random.uniform(-1,1,size=(batch_size, input_dim))

        sess.run(d_train, feed_dict={true_holder:x_batch, label_holder:label, noise_holder:noise_batch})

        d_loss_train = sess.run(d_loss, feed_dict={noise_holder:noise_batch, label_holder:label, true_holder:x_batch})
        g_loss_train = sess.run(g_loss, feed_dict={label_holder:label, noise_holder:noise_batch})
        print("重训练开始数据\ng_loss:",g_loss_train, "\td_loss:", d_loss_train)

        # # 显示图片
        # noise_test, label_onehot_test = pred_prev(0)
        # gen_img_1 = sess.run(get_generator(noise_holder, label_holder, output_shape, reuse=tf.AUTO_REUSE), feed_dict={noise_holder:noise_test, label_holder:label_onehot_test})   
        # show_img(gen_img_1)

        for epoch in range(epochs):
            batch_times = mnist.train.num_examples // batch_size
            for batch in tqdm(range(batch_times)):
                batch, label = mnist.train.next_batch(batch_size)  #一批数据（包括训练、验证图片、标签）
                x_batch = batch.reshape((batch_size, 784))  #训练图片数据
                x_batch = x_batch*2 -1
                noise_batch = np.random.uniform(-1,1,size=(batch_size, input_dim))

                _ = sess.run(d_train, feed_dict={true_holder:x_batch, label_holder:label, noise_holder:noise_batch})
                _ = sess.run(g_train, feed_dict={label_holder:label, noise_holder:noise_batch})

            d_loss_train = sess.run(d_loss, feed_dict={noise_holder:noise_batch, label_holder:label, true_holder:x_batch})
            g_loss_train = sess.run(g_loss, feed_dict={label_holder:label, noise_holder:noise_batch})
            print("重训练第", epoch, "轮epoch完毕\ng_loss:",g_loss_train, "\td_loss:", d_loss_train)

            if epoch % save_size == 0 and epoch != 0:
                saver = tf.train.Saver(var_list=g_vars)
                save_path = saver.save(sess, model_path)
                print('模型已保存在',save_path, '中\n')

                # #显示图片
                # noise_test, label_onehot_test = pred_prev(0)
                # gen_img_1 = sess.run(get_generator(noise_holder, label_holder, output_shape, reuse=tf.AUTO_REUSE), feed_dict={noise_holder:noise_test, label_holder:label_onehot_test})   
                # show_img(gen_img_1)
