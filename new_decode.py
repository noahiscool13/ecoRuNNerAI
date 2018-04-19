import scipy.io as spio
import numpy as np
import tensorflow as tf
from random import shuffle
import os
from matplotlib import pyplot as plt



batch = 100

def variable_summaries(var):
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)

mat = spio.loadmat('Outputtot2.mat', squeeze_me=True)

#print(mat['Outputtot'][0][0][1])

Outputtot = mat['Outputtot']; #Matrix met in alle output van de simulatie
speedDif = mat['speedDif']; #Vector met de mogelijke snelheidsinputs
lapNum = mat['startLap']; #Vector met mogelijke lap-inputs
lapPosList = mat['startLength']; #Vector met mopgelijke posities per lap
timeDif = mat['timeDif']; #... mogelijke tijdsinputs
lapLength = mat['lapLength']; #totale lengte van 1 lap
startLapPos = np.array((lapNum-1)*lapLength); #Beginlengte van elke lap
X,Y = np.meshgrid(startLapPos,lapPosList); #Python Magie
possiblePos = np.reshape(np.transpose((X+Y)),-1); #Alle mogelijke positites op een race die gesimuleerd zijn. Laps along x-axis, positions along y-axis.

strats = []

for racePos in possiblePos:
    for speed in speedDif:
        for time in timeDif:

            # racePos = possiblePos[0]; #Neem een bepaalde positie
            # speed = speedDif[0]; #Neem een bepaalde snelheid
            # time = timeDif[0]; #Neem een bepaalde tijd
            #print(racePos)

            lap = np.floor(float(racePos)/lapLength)+1; #Bereken met welke lap die positie overeen komt
            lapIDX = np.searchsorted(lapNum,lap); #Wat is de index van die lap
            lapPos =racePos-(lap-1)*lapLength; #Bereken met welke lengte per lap die positie overeen komt
            lengthIDX = np.searchsorted(lapPosList,lapPos); #Index van die lengte
            speedIDX = np.searchsorted(speedDif,speed); #Index van de gekozen snelheid
            timeIDX = np.searchsorted(timeDif,time); #Idx gekozen tijd

            IDX = np.size(speedDif)*np.size(timeDif)*lapIDX+np.size(timeDif)*speedIDX+timeIDX #Bepaalde index waarmee de Outputtot geordend is

            strategy = Outputtot[lengthIDX][IDX][1] #Strategie als gevolg van gekozen snelheid/tijd/positie

            #print(strategy)
            ### Nu moet dit naar nulletjes en eentjes getransformeerd worden. Voor het gemak alleen de nulletjes en eentjes van 1 ronde (de ronde waar je nu op zit). Dit zijn er al 2000...
            ### Je kan er voor kiezen om dit aantal (2000) naar 300 te verkleinen, want ten behoeve van optimalisatie van het simulatie programma, zijn er maar 300 variabel
            ### Deze variabele worden gegeven door partOne, partTwo en partThree. Je kan in principe alle andere delen verwijderen, en alleen de 300 punten die nodig zijn op te slaan.

            domain = ['startOne','lengthOne','lengthTwo','startThree','endStart', 'startTwo',  'lengthThree', 'safetyLaps','timeWindow','hillTrans','maxVolt']

            for i in range(4):
                exec(eval("domain[i]")+'=strategy[0][i]') #Python Magie
            for i in range(7):
                exec(eval("domain[i+4]")+'=strategy[1][i]') #Python Magie

            # for n in domain:
                #exec("print("+n+",n)")


            strats.append([(racePos,speed,time),(startOne,lengthOne,lengthTwo,startThree)])

            # partTotal = 20; #Aantal delen per lap
            # partLength = 100; #Aantal punten per deel
            #
            #
            # result = np.zeros((partTotal*partLength,1));
            #
            # result[int(np.floor(partLength*(startOne-1))):int(np.floor(partLength*(startOne-1)+1+lengthOne*partLength))]=1;
            # result[int(np.floor(partLength*(startTwo-1))):int(np.floor(partLength*(startTwo-1)+1+lengthTwo*partLength))]=1;
            # result[int(np.floor(partLength*(startThree-1))):int(np.floor(partLength*(startThree-1)+1+lengthThree*partLength))]=1;
            #
            # result = np.reshape(result,[partTotal,partLength]);


            #print(np.array(result)) #Dit is uiteindelijk je resultaat.

shuffle(strats)
q,a = zip(*strats)

print(len(q))

trQ = q[:int(len(q)*0.7)]
trA = a[:int(len(a)*0.7)]

trTQ = q[int(len(q)*0.7):int(len(q)*1)]
trTA = a[int(len(a)*0.7):int(len(a)*1)]

endTQ = q[int(len(q)*0.85):]
endTA = a[int(len(a)*0.85):]

x = tf.placeholder(dtype=tf.float32, shape=[None, 3])
y = tf.placeholder(dtype=tf.float32, shape=[None, 4])

learning_rate = tf.placeholder(tf.float32, shape=[])

def net(data):
    hidden_l1 = {"w": tf.Variable(tf.random_normal([3, 100])), "b": tf.Variable(tf.random_normal([100]))}
    hidden_l2 = {"w": tf.Variable(tf.random_normal([100, 500])), "b": tf.Variable(tf.random_normal([500]))}
    hidden_l3 = {"w": tf.Variable(tf.random_normal([500, 700])), "b": tf.Variable(tf.random_normal([700]))}
    hidden_l4 = {"w": tf.Variable(tf.random_normal([700, 500])), "b": tf.Variable(tf.random_normal([500]))}
    output_l = {"w": tf.Variable(tf.random_normal([500, 4])), "b": tf.Variable(tf.random_normal([4]))}

    l1 = tf.nn.sigmoid(tf.matmul(data, hidden_l1["w"]) + hidden_l1["b"])
    l2 = tf.nn.sigmoid(tf.matmul(l1, hidden_l2["w"]) + hidden_l2["b"])
    l3 = tf.nn.sigmoid(tf.matmul(l2, hidden_l3["w"]) + hidden_l3["b"])
    l4 = tf.nn.sigmoid(tf.matmul(l3, hidden_l4["w"]) + hidden_l4["b"])
    out = tf.matmul(l4, output_l["w"]) + output_l["b"]

    return out


def train_net(x, y):


    with tf.Session() as sess:
        predict = net(x)
        cost = tf.losses.mean_squared_error(predict, y)
        loss = tf.losses.absolute_difference(predict, y)
        tf.summary.scalar("loss", loss)
        merged = tf.summary.merge_all()
        test_writer = tf.summary.FileWriter('/test3')

        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        testE = []
        trainE = []

        lr = 0.1

        lr_change = {10:0.01,100:0.001,300:0.0001,800:0.00005}
        for epoch in range(300):
            #if epoch in lr_change:
                #lr = lr_change[epoch]
            for bt in range(int(len(trQ) / batch)):
                bq = trQ[bt * batch:bt * batch + batch]
                ba = trA[bt * batch:bt * batch + batch]
                _, c = sess.run([optimizer, cost], feed_dict={x: bq, y: ba, learning_rate: lr})
            # print(epoch, c)
            if epoch % 10 == 0:
                loss = tf.losses.absolute_difference(predict, y)
                loss = sess.run([loss], feed_dict={x: trQ, y: trA})
                trainE.append(loss)
                loss = tf.losses.absolute_difference(predict, y)
                loss = sess.run([loss], feed_dict={x: trTQ, y: trTA})
                testE.append(loss)
                summary = sess.run(merged, feed_dict={x: trTQ, y: trTA})
                test_writer.add_summary(summary, epoch)
                print(epoch,trainE[-1],testE[-1])

        p = sess.run([predict], feed_dict={x: trTQ, y: trTA})
        print(p[0].shape)
        a = 0
        for x in range(len(p[0])):
            a += sum([abs(p[0][x][l] - trTA[x][l]) for l in range(4)])
        print(a / p[0].shape[0] / p[0].shape[1])
        print("zzz")
        print(p[0][13])
        print("zzz")
        print(print(trTA[13]))

        saver.save(sess, os.path.join(os.getcwd(),"trained_models/NNv2_1.ckpt"))


    plt.plot(testE)
    plt.plot(trainE)
    plt.show()




train_net(x, y)