import sympy
import numpy as np
from sympy import Symbol
from functools import reduce


# вероятность распределения пуассона для значения k при параметре lambda
def getPuass(lambd, k):
    return (np.e ** (-lambd)) * (lambd ** k) / np.math.factorial(k)

# получение списка случайно распределенных чисел пуассоновскаого распределения
# start, end - интервал чисел, num - кол-во чисел, pl - параметр лямбда
def getPuassValues(start, end, num, pl):
    values = np.zeros(num)
    distrValues = np.arange(start, end, 1)

    distrProb = np.zeros(len(distrValues))
    distrProb[0] = getPuass(pl, 0)
    for i in range(1, len(distrValues)):
        distrProb[i] = distrProb[i - 1] + getPuass(pl, i)
    distrTable = dict(zip(distrProb, distrValues))
    randSamples = np.random.random_sample(num)

    for i in range(0, num):
        values[i] = distrTable[min(distrProb, key=lambda a: abs(a - randSamples[i]))]

    return values

# получение списка случайно распределенных чисел заданного распределения
# start, end -интервал чисел, num - кол-во чисел, pA, pB - параметры A и B
def getDistrValues(start, end, num, pA, pB):

    values = np.zeros(num) # список для полученных значений
    distrVal = np.linspace(start, end, num)
    x = Symbol('x')

    # вычисление рекурентной части функции распределения
    recurentPart = (1 - sympy.E ** (-pA * (x ** 2))) / pA
    for i in range(pB - 1, 0, -1):
        recurentPart = (-(pB - i + 1) / pA) * (sympy.E ** (-pA * (x ** 2)) * (x ** (2 * (pB - i))) - recurentPart)

    # функция распределения
    gamma = (2 * (pA ** (pB + 1)) / (sympy.factorial(pB)))
    distr = (-gamma / (2 * pA)) * ((x ** (2 * pB)) * (sympy.E ** (-pA * (x ** 2))) - recurentPart)

    distrProb = np.array([*map(lambda p:  distr.evalf(subs={x: p}), distrVal)])
    distrTable = dict(zip(distrProb, distrVal))  # полученная таблица
    randSamples = np.random.random_sample(num)

    # выборка случайных значений из таблицы по равномерному распределению
    for i in range(0, num):
        values[i] = distrTable[min(distrProb, key=lambda a: abs(a - randSamples[i]))]

    return values

# вычисление выборочного среднего
def getSampleAverage(values):
    varRow = toOrderedSample(values)
    return reduce(lambda a, b: a + b[0]*b[1], varRow, 0)

# вычисление выборочной дисперсии
def getSampleVariance(values, sample_avg):
    varRow = toOrderedSample(values)
    return reduce(lambda a, b: a + ((b[0] - sample_avg) ** 2) * b[1], varRow, 0)

# вычисление вариационнного ряда ([значение][вероятность])
def toOrderedSample(sample):
    ordSample = {}
    sampleProb = 1 / len(sample)
    for i in sample:
        ordSample[i] = ordSample.get(i, 0) + sampleProb
    return sorted(ordSample.items())

# вычисление интевалов для 5 задания
def getProbInterval(p, avr, samples):
    l = 0
    p_now = 0
    left = 0
    right = 0
    while p - p_now > 0:
        left = avr - l / 9
        right = avr + (5 * l / 2)
        if left < 0:
            left = 0
        t = 0
        for i in samples: # вычисление вероятности в полученном интервале
            if left <= i < right:
                t = t + 1
        p_now = t / len(samples)
        l = l + 0.01

    return left, right, l

# вычисление интервалов для 7 задания
def getVarInter(values, num):
    varInt = [] # [левая граница][правая граница]
    varValues = [] # значения в каждом интервале
    values = sorted(values)
    xMax = values[-1]
    xMin = values[0]
    intLen = abs((xMax - xMin) / num)

    for i in range(0, num):
        fstPnt = xMin + (intLen * i)
        sndPnt = fstPnt + intLen
        varInt.append([fstPnt, sndPnt])
        tmp = []
        for j in values:
            if fstPnt <= j < sndPnt:
                tmp.append(j)
        varValues.append(tmp)

    return varInt, varValues





