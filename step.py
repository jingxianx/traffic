import numpy as np
import random
from matplotlib import pyplot as plt

#Parameters
delta_t = 0.1
I = np.identity(2)
kd = 0.2
kv = 0.2
M = np.array([[0., 1.0], [-kd, -kv]])
N = np.array([[0., 0.], [kd, kv]])
K = 51
t = np.arange(0, 50, 0.1)
Y_initial = np.zeros((2, 1))

def step_info(Y):
    i = []
    overshoot = []
    for k in range(1, len(Y[0])-1):
        print Y[:, k, 0].max()
        print Y[-1, k, 0]
        print k
        overshoot.append(Y[:, k, 0].max()/Y[-1, k, 0] - 1)
        i.append(k)
    
    plt.figure(9)
    #    plt.subplot(211)
    plt.plot(i, overshoot, '-')
    plt.xlim(-1, 50)
    plt.xlabel("i")
    plt.ylabel("overshoot")
    plt.title("Overshoot")

#CFM
t = np.arange(0, 70, 0.1)
tc = np.array([[0]])
Y1 = np.array([[Y_initial for j in range(K)] for i in range(len(t))])
for n in range(1, len(Y1)):
    Y1[n, 0, 0] = 1.0
    for k in range(1, len(Y1[0])):
        Y1[n, k] = np.dot((I+np.dot(M, delta_t)), Y1[n-1, k]) + np.dot(np.dot(delta_t, N), Y1[n-1, k-1])
    tc = np.append(tc, [[n*delta_t]])
#step_info(Y1)

plt.figure(1)
plt.subplot(211)
for k in range(0, len(Y1[0])):
    plt.plot(Y1[:, k, 0]/30 + k, tc, '-')
plt.xlim(-1, 50)
plt.xlabel("yk(t) (k = 1, 2, ..., 50)")
plt.ylabel("time (second)")
plt.title("CFM Relative Position Step Response of Position Change")

plt.subplot(212)
for k in range(0, len(Y1[0])):
    plt.plot(Y1[:, k, 1]/25 + k, tc, '-')
plt.xlim(-1, 51)
plt.xlabel("dotyk(t) (k = 1, 2, ..., 50)")
plt.ylabel("time (second)")
plt.title("CFM Relative Speed Step Response of Position Change")

tc = np.array([[0]])
Y2 = np.array([[Y_initial for j in range(K)] for i in range(len(t))])
for n in range(1, len(Y2)):
    Y2[n, 0, 1] = 1.0
    for k in range(1, len(Y2[0])):
        Y2[n, k] = np.dot((I+np.dot(M, delta_t)), Y2[n-1, k]) + np.dot(np.dot(delta_t, N), Y2[n-1, k-1])
    tc = np.append(tc, [[n*delta_t]])
#step_info(Y2)

plt.figure(2)
plt.subplot(211)
for k in range(0, len(Y2[0])):
    plt.plot(Y2[:, k, 0]/30 + k, tc, '-')
plt.xlim(-1, 50)
plt.xlabel("yk(t) (k = 1, 2, ..., 50)")
plt.ylabel("time (second)")
plt.title("CFM Relative Position Step Response of Speed Change")

plt.subplot(212)
for k in range(0, len(Y2[0])):
    plt.plot(Y2[:, k, 1]/25 + k, tc, '-')
plt.xlim(-1, 51)
plt.xlabel("dotyk(t) (k = 1, 2, ..., 50)")
plt.ylabel("time (second)")
plt.title("CFM Relative Speed Step Response of Speed Change")

#BCM
tBCM = 270
t = np.arange(0, 300, 0.1)
tc = np.array([[0]])
Y3 = np.array([[Y_initial for j in range(K)] for i in range(len(t))])
for n in range(1, tBCM):
    Y3[n, 0, 0] = 1.0
    for k in range(1, len(Y3[0])):
        Y3[n, k] = np.dot((I+np.dot(M, delta_t)), Y3[n-1, k]) + np.dot(np.dot(delta_t, N), Y3[n-1, k-1])
    tc = np.append(tc, [[n*delta_t]])

for n in range(tBCM, len(Y3)):
    Y3[n, 0, 0] = 1.0
    for k in range(1, len(Y3[0]) - 1):
        Y3[n, k] = np.dot((I+np.dot(M, delta_t)), Y3[n-1, k]) + 0.5 * np.dot(np.dot(delta_t, N), Y3[n-1, k-1] + Y3[n-1, k+1])
    tc = np.append(tc, [[n*delta_t]])
step_info(Y3)

plt.figure(3)
plt.subplot(211)
for k in range(0, len(Y3[0])):
    plt.plot(Y3[:, k, 0]/30 + k, tc, '-')
plt.xlim(-1, 50)
plt.xlabel("yk(t) (k = 1, 2, ..., 50)")
plt.ylabel("time (second)")
plt.title("BCM Relative Position Step Response of Position Change")

plt.subplot(212)
for k in range(0, len(Y3[0])):
    plt.plot(Y3[:, k, 1]/25 + k, tc, '-')
plt.xlim(-1, 51)
plt.xlabel("dotyk(t) (k = 1, 2, ..., 50)")
plt.ylabel("time (second)")
plt.title("BCM Relative Speed Step Response of Position Change")

tBCM = 270
t = np.arange(0, 90, 0.1)
tc = np.array([[0]])
Y4 = np.array([[Y_initial for j in range(K)] for i in range(len(t))])
for n in range(1, tBCM):
    Y4[n, 0, 1] = 1.0
    for k in range(1, len(Y4[0])):
        Y4[n, k] = np.dot((I+np.dot(M, delta_t)), Y4[n-1, k]) + np.dot(np.dot(delta_t, N), Y4[n-1, k-1])
    tc = np.append(tc, [[n*delta_t]])

for n in range(tBCM, len(Y4)):
    Y4[n, 0, 1] = 1.0
    for k in range(1, len(Y4[0]) - 1):
        Y4[n, k] = np.dot((I+np.dot(M, delta_t)), Y4[n-1, k]) + 0.5 * np.dot(np.dot(delta_t, N), Y4[n-1, k-1] + Y4[n-1, k+1])
    tc = np.append(tc, [[n*delta_t]])

plt.figure(4)
plt.subplot(211)
for k in range(0, len(Y4[0])):
    plt.plot(Y4[:, k, 0]/30 + k, tc, '-')
plt.xlim(-1, 50)
plt.xlabel("yk(t) (k = 1, 2, ..., 50)")
plt.ylabel("time (second)")
plt.title("BCM Relative Position Step Response of Speed Change")

plt.subplot(212)
for k in range(0, len(Y4[0])):
    plt.plot(Y4[:, k, 1]/25 + k, tc, '-')
plt.xlim(-1, 51)
plt.xlabel("dotyk(t) (k = 1, 2, ..., 50)")
plt.ylabel("time (second)")
plt.title("BCM Relative Speed Step Response of Speed Change")

#Interleave
block = 4
tc = np.array([[0]])
Y5 = np.array([[Y_initial for j in range(K)] for i in range(len(t))])
for n in range(1, len(Y5)):
    Y5[n, 0, 0] = 1.0
    for k in range(1, len(Y5[0])-block, block):
        Y5[n, k] = np.dot((I+np.dot(M, delta_t)), Y5[n-1, k]) + np.dot(np.dot(delta_t, N), Y5[n-1, k-1])
        for b in range(block-1):
            Y5[n, k+b+1] = np.dot((I+np.dot(M, delta_t)), Y5[n-1, k+b+1]) + 0.5 * np.dot(np.dot(delta_t, N), Y5[n-1, k+b] + Y5[n-1, k+b+2])
    tc = np.append(tc, [[n*delta_t]])

plt.figure(5)
plt.subplot(211)
for k in range(0, len(Y5[0])):
    plt.plot(Y5[:, k, 0]/30 + k, tc, '-')
plt.xlim(-1, 50)
plt.xlabel("yk(t) (k = 1, 2, ..., 50)")
plt.ylabel("time (second)")
plt.title("Interleave Relative Position Step Response of Position Change")

plt.subplot(212)
for k in range(0, len(Y5[0])):
    plt.plot(Y5[:, k, 1]/25 + k, tc, '-')
plt.xlim(-1, 51)
plt.xlabel("dotyk(t) (k = 1, 2, ..., 50)")
plt.ylabel("time (second)")
plt.title("Interleave Relative Speed Step Response of Position Change")

tc = np.array([[0]])
Y6 = np.array([[Y_initial for j in range(K)] for i in range(len(t))])
for n in range(1, len(Y6)):
    Y6[n, 0, 1] = 1.0
    for k in range(1, len(Y6[0])-block, block):
        Y6[n, k] = np.dot((I+np.dot(M, delta_t)), Y6[n-1, k]) + np.dot(np.dot(delta_t, N), Y6[n-1, k-1])
        for b in range(block-1):
            Y6[n, k+b+1] = np.dot((I+np.dot(M, delta_t)), Y6[n-1, k+b+1]) + 0.5 * np.dot(np.dot(delta_t, N), Y6[n-1, k+b] + Y6[n-1, k+b+2])
    tc = np.append(tc, [[n*delta_t]])

plt.figure(6)
plt.subplot(211)
for k in range(0, len(Y6[0])):
    plt.plot(Y6[:, k, 0]/30 + k, tc, '-')
plt.xlim(-1, 50)
plt.xlabel("yk(t) (k = 1, 2, ..., 50)")
plt.ylabel("time (second)")
plt.title("Interleave Relative Position Step Response of Speed Change")

plt.subplot(212)
for k in range(0, len(Y6[0])):
    plt.plot(Y6[:, k, 1]/25 + k, tc, '-')
plt.xlim(-1, 51)
plt.xlabel("dotyk(t) (k = 1, 2, ..., 50)")
plt.ylabel("time (second)")
plt.title("Interleave Relative Speed Step Response of Speed Change")

#Probabilistic
p = 0.7
tc = np.array([[0]])
Y7 = np.array([[Y_initial for j in range(K)] for i in range(len(t))])
for n in range(1, len(Y7)):
    Y7[n, 0, 0] = 1.0
    for k in range(1, len(Y7[0])-1):
        dice = random.random()
        if dice < p:
            Y7[n, k] = np.dot((I+np.dot(M, delta_t)), Y7[n-1, k]) + 0.5 * np.dot(np.dot(delta_t, N), Y7[n-1, k-1] + Y7[n-1, k+1])
        else:
            Y7[n, k] = np.dot((I+np.dot(M, delta_t)), Y7[n-1, k]) + np.dot(np.dot(delta_t, N), Y7[n-1, k-1])
    tc = np.append(tc, [[n*delta_t]])
plt.figure(7)
plt.subplot(211)
for k in range(0, len(Y7[0])):
    plt.plot(Y7[:, k, 0]/30 + k, tc, '-')
plt.xlim(-1, 50)
plt.xlabel("yk(t) (k = 1, 2, ..., 50)")
plt.ylabel("time (second)")
plt.title("Interleave Relative Position Step Response of Position Change")

plt.subplot(212)
for k in range(0, len(Y7[0])):
    plt.plot(Y7[:, k, 1]/25 + k, tc, '-')
plt.xlim(-1, 51)
plt.xlabel("dotyk(t) (k = 1, 2, ..., 50)")
plt.ylabel("time (second)")
plt.title("Interleave Relative Speed Step Response of Position Change")

tc = np.array([[0]])
Y8 = np.array([[Y_initial for j in range(K)] for i in range(len(t))])
for n in range(1, len(Y8)):
    Y8[n, 0, 1] = 1.0
    for k in range(1, len(Y8[0])-1):
        dice = random.random()
        if dice < p:
            Y8[n, k] = np.dot((I+np.dot(M, delta_t)), Y8[n-1, k]) + 0.5 * np.dot(np.dot(delta_t, N), Y8[n-1, k-1] + Y8[n-1, k+1])
        else:
            Y8[n, k] = np.dot((I+np.dot(M, delta_t)), Y8[n-1, k]) + np.dot(np.dot(delta_t, N), Y8[n-1, k-1])
    tc = np.append(tc, [[n*delta_t]])
plt.figure(8)
plt.subplot(211)
for k in range(0, len(Y8[0])):
    plt.plot(Y8[:, k, 0]/30 + k, tc, '-')
plt.xlim(-1, 50)
plt.xlabel("yk(t) (k = 1, 2, ..., 50)")
plt.ylabel("time (second)")
plt.title("Interleave Relative Position Step Response of Speed Change")

plt.subplot(212)
for k in range(0, len(Y8[0])):
    plt.plot(Y8[:, k, 1]/25 + k, tc, '-')
plt.xlim(-1, 51)
plt.xlabel("dotyk(t) (k = 1, 2, ..., 50)")
plt.ylabel("time (second)")
plt.title("Interleave Relative Speed Step Response of Speed Change")
plt.show()

