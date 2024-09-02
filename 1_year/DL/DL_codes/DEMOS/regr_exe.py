#linear regression example
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import time

#a bunch of points on the plain
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([14,12,13,15,11,9,8,4,2,1])

#gradient of the quadratic loss
def grad(a,b):
    d = y - (a*x + b)      #derivative of the loss
    da = - np.sum(d * x)   #derivative of d w.r.t. a
    db = - np.sum(d)       #derivative of d w.r.t. b 
    return(da,db)

lr = 0.001
epochs = 1000

#step 1
a = np.random.rand()
b = np.random.rand()
params=[a,b]

fig = plt.figure()
plt.plot(x,y,'ro')
line, = plt.plot([], [], lw=2)

def init():
    #current approximation
    line.set_data([x[0],x[9]],[a*x[0]+b,a*x[9]+b])
    return line,

def step(i):
    a,b=params
    da,db = grad(a,b)
    print("current loss = {}".format(np.sum((y-a*x-b)**2)))
    params[0] = a - lr*da
    params[1] = b - lr*db
    ##### for animation
    line.set_data([x[0],x[9]],[a*x[0]+b,a*x[9]+b])
    #time.sleep(.01)
    return line,

anim = anim.FuncAnimation(fig, step, init_func=init, frames=2500, interval=1, blit=True, repeat=False)

plt.savefig("regr_exe.jpg")
plt.show()
