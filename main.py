import math

import numpy as np
from matplotlib import pyplot as plt

#generowanie punktow
def generatePoints(minimum, maximum, samples):
    minimum = int(minimum)
    maximum = int(maximum)
    samplesNum = int(samples)
    return np.random.uniform(minimum, maximum, samplesNum)

#rysowanie funkcji reLU
def relu(points):
    f = plt.figure(1)
    for i in points:
        valueOfFunction = max(0, i)
        plt.plot(i, valueOfFunction, 'bo')
        plt.title("Funkcja reLU")
    f.show()

#rysowanie sigmoidu
def sigmoid(points):
    g = plt.figure(2)
    for i in points:
        valueOfFunction = 1 / (1 + math.e ** i)
        plt.plot(i, valueOfFunction, 'ro')
        plt.title("Funkcja sigmoid")
    g.show()

#rysowanie funkcji tanh
def tanh(points):
    z = plt.figure(2)
    for i in points:
        valueOfFunction = (math.e ** i - math.e ** (-i)) / (math.e ** i + math.e ** (-i))
        plt.plot(i, valueOfFunction, 'go')
        plt.title("Funkcja tanh")
    z.show()

#generuje punktu od -5,5 a nastepnie rysuje kolejne wykresy
def taskOne():
    a = -5
    b = 5
    samples = 100
    generatedPoints = generatePoints(a, b, samples)
    print(generatedPoints)
    relu(generatedPoints)
    sigmoid(generatedPoints)
    tanh(generatedPoints)


# taskOne()

#korzystam z wzoru na softmax, i obliczam przynaleznosc do klastra
def clasterPossibility(points):
    denominator = 0
    for i in points:
        denominator += math.e ** i

    for i in points:
        value = (math.e ** i) / denominator
        print(str(i) + ": " + str(value))

clasterPossibility(([5,2,3,1]))