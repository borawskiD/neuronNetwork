import math
import numpy as np


# skopoiowana funkcja ze skryptu, wyszla mi funkcja numer 6
def index(num):
    sum = 0
    while num > 0:
        sum += num % 10
        num = int(num / 10)
    if sum < 10:
        return sum
    return index(sum)


numer_indeksu = 24000


# implementacja funkcji ktora wylosowalem
def f(x, y):
    return (math.sqrt(math.sin(x))) + 2 * y


# przeliczanie roznicy pomiedzy prawidlowa odpowiedzia a odpowiedzia sieci neuronowej
def loss(y_pred, y_true):
    return y_pred - y_true


# prosta funkcja do generowania n punktow z zakresu [min,max] :-)
def generatePoints(n, min, max):
    points = [np.random.uniform(min, max, n), np.random.uniform(min, max, n)]
    return points


# ilosc punktow, ktore uznajemy za wiedze i wykorzystujemy do uczenia sieci neuronowej
knowledge_size = 20
# ilosc cykli nauczania
iterations = 50
# zmienna do petli zewnetrznej
counter = 0
# ilosc neuronow wejsciowych, nie zmieniam wartosci - dalej sa dwie zmienne jako input
neurons_in = 2
# ilosc ukrytych neuronow, zmienilem z 2 na pięć bo danych jest zdecydowanie wiecej
neurons_hidden = 5
# neurony wyjsciowe bez zmian - to wciaz 1 wartosc
neurons_out = 1
# zapelniam wiedze losowymi punktami
knowledge = generatePoints(knowledge_size, 0, 3)
# to tez bez zmian
learning_rate = 0.01
# inicjacja zmiennych
W1 = np.random.random([neurons_in, neurons_hidden])
W2 = np.random.random([neurons_hidden, neurons_out])
# petla zewnetrzna odpowiada za wyuczanie calej sieci okreslona ilosc razy
while counter < iterations - 1:
    print("\n Iteration number: " + str(counter))
    a = 0
    # petla wewnetrzna natomiast odpowiada za to, zeby za kazdym razem nauczanie przeszlo przez kazdy punkt bedacy w bazie wiedzy
    while a < knowledge_size:
        # definiujemy x, y prawidlowe
        X = np.array([knowledge[0][a], knowledge[1][a]])
        y_true = f(knowledge[0][a], knowledge[1][a])
        # liczymy wektor H i przewidywana wartosc funkcji przez SI
        H = np.dot((X), W1)
        y_pred = np.dot(H, W2)
        # liczymy wartosc bledu
        er = loss(y_pred, y_true)
        # i nowa wartosc wektorow W2 i W1
        W2 = W2 - (learning_rate * er * H.reshape(-1, 1))
        W1 = W1 - (learning_rate * er * X.reshape(-1, 1) * W2.T)
        print("Input: x = " + str(knowledge[0][a]) + " y = " + str(knowledge[0][a]) + " correct answer: " + str(
            y_true) + " AI answer: " + str(y_pred))
        a += 1

    counter += 1

print("end of loop")
