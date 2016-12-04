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

tc = np.array([[0]])
Y1 = np.array([[Y_initial for j in range(K)] for i in range(len(t))])
for n in range(1, len(Y1)):
    for k in range(1, len(Y1[0])):
        Y1[n, k] = np.dot((I+np.dot(M, delta_t)), Y1[n-1, k]) + np.dot(np.dot(delta_t, N), Y1[n-1, k-1])
    tc = np.append(tc, [[n*delta_t]])

plt.figure(1)
plt.subplot(211)
for k in range(0, len(Y1[0])):
    plt.plot(Y1[:, k, 0]/30 + k, tc, '-')
plt.xlim(-1, 51)
plt.xlabel("yk(t) (k = 1, 2, ..., 50)")
plt.ylabel("time (second)")
plt.title("CFM Relative Position without Perturbations")

plt.subplot(212)
for k in range(0, len(Y1[0])):
    plt.plot(Y1[:, k, 1]/25 + k, t, '-')
plt.xlim(-1, 51)
plt.xlabel("dotyk(t) (k = 1, 2, ..., 50)")
plt.ylabel("time (second)")
plt.title("CFM Relative Speed without Perturbations")

tc = np.array([[0]])
Y2 = np.array([[Y_initial for j in range(K)] for i in range(len(t))])
for n in range(1, len(Y2)):
    for k in range(1, len(Y2[0])):
        Y2[n, k] = np.dot((I+np.dot(M, delta_t)), Y2[n-1, k]) + np.dot(np.dot(delta_t, N), Y2[n-1, k-1])
        if n == 1:
            Y2[n, k, 0] += random.uniform(-1.5, 1.5)
    tc = np.append(tc, [[n*delta_t]])

plt.figure(2)
plt.subplot(211)
for k in range(0, len(Y2[0])):
    plt.plot(Y2[:, k, 0]/30 + k, tc, '-')
plt.xlim(-1, 50)
plt.xlabel("yk(t) (k = 1, 2, ..., 50)")
plt.ylabel("time (second)")
plt.title("CFM Relative Position with Perturbations in Positions")

plt.subplot(212)
for k in range(0, len(Y2[0])):
    plt.plot(Y2[:, k, 1]/25 + k, tc, '-')
plt.xlim(-1, 51)
plt.xlabel("dotyk(t) (k = 1, 2, ..., 50)")
plt.ylabel("time (second)")
plt.title("CFM Relative Speed with Perturbations in Positions")

tc = np.array([[0]])
Y3 = np.array([[Y_initial for j in range(K)] for i in range(len(t))])
for n in range(1, len(Y3)):
    for k in range(1, len(Y3[0])):
        Y3[n, k] = np.dot((I+np.dot(M, delta_t)), Y3[n-1, k]) + np.dot(np.dot(delta_t, N), Y3[n-1, k-1])
        if n == 1:
            Y3[n, k, 1] += random.uniform(-1.0, 1.0)
    tc = np.append(tc, [[n*delta_t]])

plt.figure(3)
plt.subplot(211)
for k in range(0, len(Y3[0])):
    plt.plot(Y3[:, k, 0]/30 + k, tc, '-')
plt.xlim(-1, 50)
plt.xlabel("yk(t) (k = 1, 2, ..., 50)")
plt.ylabel("time (second)")
plt.title("CFM Relative Position with Perturbations in Speed")

plt.subplot(212)
for k in range(0, len(Y3[0])):
    plt.plot(Y3[:, k, 1]/25 + k, tc, '-')
plt.xlim(-1, 51)
plt.xlabel("dotyk(t) (k = 1, 2, ..., 50)")
plt.ylabel("time (second)")
plt.title("CFM Relative Speed with Perturbations in Speed")

t = np.arange(0, 70, 0.1)
tc = np.array([[0]])
Y4 = np.array([[Y_initial for j in range(K)] for i in range(len(t))])
for n in range(1, len(Y4)):
    Y4[n, 0, 1] += random.uniform(-1.0, 1.0)
    for k in range(1, len(Y4[0])):
        Y4[n, k] = np.dot((I+np.dot(M, delta_t)), Y4[n-1, k]) + np.dot(np.dot(delta_t, N), Y4[n-1, k-1])
    tc = np.append(tc, [[n*delta_t]])

plt.figure(4)
plt.subplot(211)
for k in range(0, len(Y4[0])):
    plt.plot(Y4[:, k, 0]/30 + k, tc, '-')
plt.xlim(-1, 50)
plt.xlabel("yk(t) (k = 1, 2, ..., 50)")
plt.ylabel("time (second)")
plt.title("CFM Relative Position with Perturbations in Speed of the 1st Car")

plt.subplot(212)
for k in range(0, len(Y4[0])):
    plt.plot(Y4[:, k, 1]/25 + k, tc, '-')
plt.xlim(-1, 51)
plt.xlabel("dotyk(t) (k = 1, 2, ..., 50)")
plt.ylabel("time (second)")
plt.title("CFM Relative Speed with Perturbations in Speed of the 1st Car")
plt.show()

