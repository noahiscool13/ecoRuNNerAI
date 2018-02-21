import tensorflow as tf
import os
import skimage.data
import numpy as np
import matplotlib.pyplot as plt 
import PIL
from skimage import transform 
from skimage.color import rgb2gray

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = "/home/noahiscool13/Documents/TF"
train_data_directory = os.path.join(ROOT_PATH, "Training")
test_data_directory = os.path.join(ROOT_PATH, "Testing")

images, labels = load_data(train_data_directory)
Timages, Tlabels = load_data(test_data_directory)

n = []
for x in labels:
    t = [0]*62
    t[x]=1
    n.append(t)
labels = n

n = []
for x in Tlabels:
    t = [0]*62
    t[x]=1
    n.append(t)
Tlabels = n


resized = [transform.resize(np.array(img),(30,30)) for img in images]
Tresized = [transform.resize(np.array(img),(30,30)) for img in Timages]


gray = [(rgb2gray(np.array(x))*255).flatten() for x in resized]
Tgray = [(rgb2gray(np.array(x))*255).flatten() for x in Tresized]

print(Tgray)

x = tf.placeholder(dtype=tf.float32, shape=[None, 900])
y = tf.placeholder(dtype=tf.int32, shape=[None,62])

def net(data):
    hidden_l1 = {"w": tf.Variable(tf.random_normal([900,900])), "b": tf.Variable(tf.random_normal([900]))}
    hidden_l2 = {"w": tf.Variable(tf.random_normal([900, 500])), "b": tf.Variable(tf.random_normal([500]))}
    hidden_l3 = {"w": tf.Variable(tf.random_normal([500, 500])), "b": tf.Variable(tf.random_normal([500]))}
    hidden_l4 = {"w": tf.Variable(tf.random_normal([500, 200])), "b": tf.Variable(tf.random_normal([200]))}
    output_l = {"w": tf.Variable(tf.random_normal([200, 62])), "b": tf.Variable(tf.random_normal([62]))}

    l1 = tf.nn.relu(tf.matmul(data,hidden_l1["w"])+hidden_l1["b"])
    l2 = tf.nn.relu(tf.matmul(l1, hidden_l2["w"]) + hidden_l2["b"])
    l3 = tf.nn.relu(tf.matmul(l2, hidden_l3["w"]) + hidden_l3["b"])
    l4 = tf.nn.relu(tf.matmul(l3, hidden_l4["w"]) + hidden_l4["b"])
    out = tf.matmul(l4, output_l["w"]) + output_l["b"]

    return out

def train_net(x,y):
    predict = net(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        testE = []
        trainE = []
        for epoch in range(100):
            _,c = sess.run([optimizer,cost],feed_dict={x:gray,y:labels})
            # print(epoch, c)
            if epoch%10 == 0:
                acuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(predict,1),tf.arg_max(y,1)),'float'))
                acuracy = acuracy.eval({x:Tgray,y:Tlabels})
                testE.append(acuracy)
                acuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(predict, 1), tf.arg_max(y, 1)), 'float'))
                acuracy = acuracy.eval({x: gray, y: labels})
                trainE.append(acuracy)
                print("epoch:",epoch,"TestAcc:",testE[-1],"TrainAcc:",trainE[-1])
    plt.plot(trainE)
    plt.plot(testE)
    plt.show()

        

train_net(x,y)

