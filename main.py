import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE  # TSNE集成在了sklearn中
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D  # 进行3D图像绘制

import input_data  # MNIST的数据操作文件

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
saver = tf.train.import_meta_graph('model/model.ckpt.meta')  # tensorflow加载神经网络图结构
gragh = tf.get_default_graph()

image_input = gragh.get_tensor_by_name('Placeholder:0')  # 获得图中预定义的输入，即MNIST图像
label_input = gragh.get_tensor_by_name('Placeholder_1:0')  # 获得对应图像的标签
predict = gragh.get_tensor_by_name('fco/BiasAdd:0')  # 获得网络的输出值

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver.restore(sess, tf.train.latest_checkpoint("model"))  # tensorflow恢复神经网络参数到当前图

    # 方便快速计算，只取训练集前面2000个数据进行可视化。
    pre = sess.run(predict,
                   feed_dict={image_input: mnist.test.images[:2000, :], label_input: mnist.test.labels[:2000, :]})

    # TSNE进行降维计算，n_components代表降维维度
    embedded = TSNE(n_components=3).fit_transform(pre)

    # 对数据进行归一化操作
    x_min, x_max = np.min(embedded, 0), np.max(embedded, 0)
    embedded = embedded / (x_max - x_min)

    # 创建显示的figure
    fig = plt.figure()
    ax = Axes3D(fig)
    # 将数据对应坐标输入到figure中，不同标签取不同的颜色，MINIST共0-9十个手写数字
    ax.scatter(embedded[:, 0], embedded[:, 1], embedded[:, 2],
               c=plt.cm.Set1(np.argmax(mnist.test.labels[:2000, :], axis=1) / 10.0))

    # 关闭了plot的坐标显示
    plt.axis('off')
    plt.show()
