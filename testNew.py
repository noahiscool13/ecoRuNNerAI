import os
import tensorflow as tf
import numpy as np

n = tf.placeholder(dtype=tf.float32, shape=[None, 4])

hidden_l1 = {"w": tf.Variable(tf.random_normal([4, 100])), "b": tf.Variable(tf.random_normal([100]))}
hidden_l2 = {"w": tf.Variable(tf.random_normal([100, 500])), "b": tf.Variable(tf.random_normal([500]))}
hidden_l3 = {"w": tf.Variable(tf.random_normal([500, 700])), "b": tf.Variable(tf.random_normal([700]))}
hidden_l4 = {"w": tf.Variable(tf.random_normal([700, 500])), "b": tf.Variable(tf.random_normal([500]))}
output_l = {"w": tf.Variable(tf.random_normal([500, 4])), "b": tf.Variable(tf.random_normal([4]))}

def net(data):
    l1 = tf.nn.sigmoid(tf.matmul(data, hidden_l1["w"]) + hidden_l1["b"])
    l2 = tf.nn.sigmoid(tf.matmul(l1, hidden_l2["w"]) + hidden_l2["b"])
    l3 = tf.nn.sigmoid(tf.matmul(l2, hidden_l3["w"]) + hidden_l3["b"])
    l4 = tf.nn.sigmoid(tf.matmul(l3, hidden_l4["w"]) + hidden_l4["b"])
    out = tf.matmul(l4, output_l["w"]) + output_l["b"]

    return out

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, os.path.join(os.getcwd(), "trained_models/NNv2_1.ckpt"))

pred = net(n)

def run_net(data):
    return sess.run(pred,feed_dict={n:np.array([data])})

def close():
    sess.close()