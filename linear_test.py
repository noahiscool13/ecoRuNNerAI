import tensorflow as tf
import os
import skimage.data
import numpy as np
import matplotlib.pyplot as plt
import PIL
from skimage import transform
from skimage.color import rgb2gray

batch = 100

q = []
a = []

for x in range(10):
    for y in range(100):
        q.append([x*10+y/10])
        t = [0]*10
        t[x] = 1
        a.append(t)

# q = np.array(q)
# a = np.array(a)

print(q)

x = tf.placeholder(dtype=tf.float32, shape=[None, 1])
y = tf.placeholder(dtype=tf.int32, shape=[None, 10])


def net(data):
    hidden_l1 = {"w": tf.Variable(tf.random_normal([1, 30])), "b": tf.Variable(tf.random_normal([30]))}
    #hidden_l2 = {"w": tf.Variable(tf.random_normal([50, 20])), "b": tf.Variable(tf.random_normal([20]))}
    output_l = {"w": tf.Variable(tf.random_normal([30, 10])), "b": tf.Variable(tf.random_normal([10]))}

    l1 = tf.nn.relu(tf.matmul(data, hidden_l1["w"]) + hidden_l1["b"])
    #l2 = tf.nn.relu(tf.matmul(l1, hidden_l2["w"]) + hidden_l2["b"])
    out = tf.matmul(l1, output_l["w"]) + output_l["b"]

    return out


def train_net(x, y):
    predict = net(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        testE = []
        trainE = []
        for epoch in range(4000):
            for bt in range(int(len(q)/batch)):
                bq = q[bt*batch:bt*batch+batch]
                ba = a[bt * batch:bt * batch + batch]
                _, c = sess.run([optimizer, cost], feed_dict={x: bq, y: ba})
            # print(epoch, c)
            if epoch % 10 == 0:
                acuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(predict, 1), tf.arg_max(y, 1)), 'float'))
                acuracy = acuracy.eval({x: q, y: a})
                testE.append(acuracy)
                acuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(predict, 1), tf.arg_max(y, 1)), 'float'))
                acuracy = acuracy.eval({x: q, y: a})
                trainE.append(acuracy)
                print("epoch:", epoch, "TestAcc:", testE[-1], "TrainAcc:", trainE[-1])
    plt.plot(trainE)
    plt.plot(testE)
    plt.show()


train_net(x, y)

