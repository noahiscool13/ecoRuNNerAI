import testNew
from random import random
import time
import matplotlib.pyplot as plt
s = []

l = time.time()

a = time.time()
for x in range(10**3):
    #print(x)
    dp = [random(),random(),random(),random()]
    q = testNew.run_net(dp)

    s.append(time.time()-l)
    l = time.time()
    #print(q)

print((time.time()-a)/10**5)
plt.plot(s)
plt.show()