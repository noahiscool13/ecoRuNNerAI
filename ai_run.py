import scipy.io as spio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from random import shuffle
import os
import tqdm

import time

batch = 100

def variable_summaries(var):
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)

def calc(racePos, speed, time):
    lap = np.floor(float(racePos) / lapLength)+1  # Bereken met welke lap die positie overeen komt
    lapIDX = np.searchsorted(lapNum, lap)  # Wat is de index van die lap
    lapPos = round(racePos - (lap - 1) * lapLength)  # Bereken met welke lengte per lap die positie overeen komt
    lengthIDX = np.searchsorted(lapPosList, lapPos)  # Index van die lengte
    speedIDX = np.searchsorted(speedDif, speed)  # Index van de gekozen snelheid
    timeIDX = np.searchsorted(timeDif, time)  # Idx gekozen tijd

    IDX = np.size(speedDif) * np.size(timeDif) * lapIDX + np.size(
        timeDif) * speedIDX + timeIDX  # Bepaalde index waarmee de Outputtot geordend is

    strategy = Outputtot[lengthIDX][IDX][1]  # Strategie als gevolg van gekozen snelheid/tijd/positie

    ### Nu moet dit naar nulletjes en eentjes getransformeerd worden. Voor het gemak alleen de nulletjes en eentjes van 1 ronde (de ronde waar je nu op zit). Dit zijn er al 2000...
    ### Je kan er voor kiezen om dit aantal (2000) naar 300 te verkleinen, want ten behoeve van optimalisatie van het simulatie programma, zijn er maar 300 variabel
    ### Deze variabele worden gegeven door partOne, partTwo en partThree. Je kan in principe alle andere delen verwijderen, en alleen de 300 punten die nodig zijn op te slaan.

    domain = ['startOne', 'endTwo', 'startThree', 'endStart', 'partOne', 'endOne', 'partTwo', 'startTwo', 'partThree',
              'endThree', 'partFour', 'startFour', 'endFour', 'partFive', 'startFive', 'endFive', 'safetyLaps',
              'timeWindow', 'hillTrans']

    dom = {}
    for i in range(0, np.size(strategy)):
        dom[domain[i]] = strategy[i]
        #exec(eval("domain[i]") + '=strategy[i]')  # Python Magie

    partTotal = 3;  # Aantal delen per lap
    partLength = 100;  # Aantal punten per deel

    result = np.zeros((partTotal,
                       partLength));  # Initialiseer array ##Kan je een 3x100 array van maken moet je wel ff zorgen dat het verder initialiseren ervan goed gaat

    # Bereken de startpunten per deel, aanvankelijk van de variabele

    startOnePoint = int(dom["startOne"] * partLength);
    endOnePoint = int(dom["endOne"] * partLength);

    startTwoPoint = int(dom["startTwo"] * partLength);
    endTwoPoint = int(dom["endTwo"] * partLength);

    startThreePoint = int(dom["startThree"] * partLength);
    endThreePoint = int(dom["endThree"] * partLength);

    result[int(1) - 1, startOnePoint:] = 1;
    result[int(1) - 1, endOnePoint:] = 0;
    result[int(2) - 1, startTwoPoint:] = 1;
    result[int(2) - 1, endTwoPoint:] = 0;
    result[int(3) - 1, startThreePoint:] = 1;
    result[int(3) - 1, endThreePoint:] = 0;

    return np.array(result)  # Dit is uiteindelijk je resultaat.




mat = spio.loadmat('OutputLs1.mat', squeeze_me=True)

Outputtot = mat['Outputtot'];  # Matrix met in alle output van de simulatie
speedDif = mat['speedDif'];  # Vector met de mogelijke snelheidsinputs
lapNum = mat['startLap'];  # Vector met mogelijke lap-inputs
lapPosList = mat['startLength'];  # Vector met mopgelijke posities per lap
timeDif = mat['timeDif'];  # ... mogelijke tijdsinputs
lapLength = mat['lapLength'];  # totale lengte van 1 lap

startLapPos = np.array((lapNum - 1) * lapLength);  # Beginlengte van elke lap
X, Y = np.meshgrid(startLapPos, lapPosList);  # Python Magie

possiblePos = np.reshape(np.transpose((X + Y)),
                         -1);  # Alle mogelijke positites op een race die gesimuleerd zijn. Laps along x-axis, positions along y-axis.



data = {}

q = []
a = []

for x in speedDif:
    for y in timeDif:
        for z in possiblePos:
            data[(z,x,y)] = calc(z,x,y).flatten()
            q.append((z,x,y))
            a.append(calc(z,x,y).flatten())

c = list(zip(q,a))

shuffle(c)

q,a = zip(*c)

trQ = q[:int(len(q)*0.7)]
trA = a[:int(len(a)*0.7)]

trTQ = q[int(len(q)*0.7):int(len(q)*1)]
trTA = a[int(len(a)*0.7):int(len(a)*1)]

endTQ = q[int(len(q)*0.85):]
endTA = a[int(len(a)*0.85):]


tf.reset_default_graph()

hidden_l1 = {"w": tf.Variable(tf.random_normal([3, 100])), "b": tf.Variable(tf.random_normal([100]))}
hidden_l2 = {"w": tf.Variable(tf.random_normal([100, 500])), "b": tf.Variable(tf.random_normal([500]))}
hidden_l3 = {"w": tf.Variable(tf.random_normal([500, 700])), "b": tf.Variable(tf.random_normal([700]))}
hidden_l4 = {"w": tf.Variable(tf.random_normal([700, 500])), "b": tf.Variable(tf.random_normal([500]))}
output_l = {"w": tf.Variable(tf.random_normal([500, 300])), "b": tf.Variable(tf.random_normal([300]))}

def net(data):


    l1 = tf.nn.sigmoid(tf.matmul(data, hidden_l1["w"]) + hidden_l1["b"])
    l2 = tf.nn.sigmoid(tf.matmul(l1, hidden_l2["w"]) + hidden_l2["b"])
    l3 = tf.nn.sigmoid(tf.matmul(l2, hidden_l3["w"]) + hidden_l3["b"])
    l4 = tf.nn.sigmoid(tf.matmul(l3, hidden_l4["w"]) + hidden_l4["b"])
    out = tf.matmul(l4, output_l["w"]) + output_l["b"]

    return out

saver = tf.train.Saver()

with tf.Session() as sess:
    completed = 0
    saver.restore(sess, os.path.join(os.getcwd(), "trained_models/NN1.ckpt"))
    totalT = []
    dataE = trQ[0]
    p = net([dataE])
    p.eval()
    for x in tqdm.tqdm(range(1000)):
        ts = time.time()
        dataE = trQ[x]
        p.eval()
        totalT.append(time.time()-ts)
    #print(p)
    plt.plot(totalT)
    plt.show()