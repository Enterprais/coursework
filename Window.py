from Design_py import window
from PyQt5 import (QtWidgets, QtCore)
import ProbFunc as pf
from PyQt5.QtWidgets import QTableWidgetItem
import matplotlib.pyplot as plt
import copy
import numpy as np


class MainWindow(QtWidgets.QWidget, window.Ui_Form):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.msgBox = QtWidgets.QMessageBox()
        self .setFixedSize(self.width(), self .height())
        self.start1_b.clicked.connect(self.pushButtonPart1)
        self.start2_b.clicked.connect(self.pushButtonPart2)
        self.start1_45_b.clicked.connect(self.pushButtonPart1_45)
        self.plot_b.clicked.connect(self.showPlot)
        self.num_of_val = 0
        self.param_A = 0
        self.param_B = 0
        self.param_lambda = 0
        self.step = 0
        self.distr_values = []
        self.puass_values = []
        self.sampAverDistr = 0
        self.sampAverPuass = 0
        self.sampVarDisrt = 0
        self.sampVarPuass = 0
        self.num_of_int = 0
        self.data_values = []
        self.varInt = []
        self.varValues = []
        self.hi2_prob = {0.005: 1, 0.01: 2, 0.025: 3, 0.05: 4, 0.1: 5, 0.25: 6, 0.5: 7}
        self.student_prob = {0.01: 1, 0.05: 2, 0.1: 3, 0.15: 4, 0.2: 5, 0.25: 6, 0.3: 7}
        for i in self.student_prob:
            self.p5_cb.addItem(str(i*2))
        for i in self.hi2_prob:
            self.p6_cb.addItem(str(i))
        self.param_gamma = 0
        self.laplas_table = []
        self.student_table = []
        self.hi2_table = []
        self.future_prob = 0
        self.valuesReady(False)
        self.valuesReadySecond(False)

    def showMessage(self, mes):
        self.msgBox.setText(mes)
        self.msgBox.exec()

    def valuesReady(self, ready: bool):
        self.start1_45_b.setEnabled(ready)

    def valuesReadySecond(self, ready: bool):
        self.plot_b.setEnabled(ready)

    # заполнение таблиц значений 1 задания 1 части
    def fillTableFirst(self, table, values):
        table.clear()
        table.setColumnCount(1)
        table.setRowCount(0)
        table.setHorizontalHeaderLabels(["Значения"])
        for i in range(0, len(values)):
            table.insertRow(table.rowCount())
            table.setItem(table.rowCount() - 1, 0, QTableWidgetItem(str(f'{values[i]:.4f}')))

    # проверка введенных параметров для расчета  задания 1 части
    def checkInputFirst(self):
        try:
            self.num_of_val = int(self.number_l.text())
            if self.num_of_val <= 10:
                raise Exception
        except Exception:
            self.showMessage("Кол-во чисел - целое число больше 10")
            return False

        try:
            self.param_A = float(self.param_A_l.text())
            if self.param_A <= 0:
                raise Exception
        except Exception:
            self.showMessage("Параметр А - число больше 0")
            return False

        try:
            self.param_B = int(self.param_s_l.text())
            if self.param_B <= 0:
                raise Exception
        except Exception:
            self.showMessage("Параметр B - целое число больше 0")
            return False

        try:
            self.param_lambda = float(self.param_lamd_l.text())
            if self.param_lambda <= 0:
                raise Exception
        except Exception:
            self.showMessage("Параметр lambda - число больше 0")
            return False

        return True

    # проверка введенных параметров для расчета  задания 2 части
    def checkInputSecond(self):
        try:
            self.num_of_int = int(self.interval_l.text())
            if self.num_of_int < 2 or self.num_of_int > 33:
                raise Exception
        except Exception:
            self.showMessage("Кол-во интервалов - от 2 до 33")
            return False

        try:
            self.param_gamma = float(self.p4_2_l.text())
            if self.param_gamma < 0 or self.param_gamma > 1:
                raise Exception
        except Exception:
            self.showMessage("Заданная вероятность гамма - число от 0 до 1")
            return False

        lines = []
        try:
            with open('data.txt', 'r') as f:
                lines = str.split(''.join(f.readlines()))
        except IOError as e:
            self.showMessage("Файл с данными не найден")
            return False
        try:
            self.data_values = [*(map(lambda x: int(x), lines))]
        except Exception as e:
            self.showMessage("Некорректный файл данных")
            return False

        try:
            with open('laplas.txt', 'r') as f:
                lines = f.readlines()
        except IOError as e:
            self.showMessage("Файл с таблицей Лапласа не найден")
            return False
        try:
            self.laplas_table = []
            for i in lines:
                self.laplas_table.append([*(map(float, i.strip('\n').split(';')))])
        except Exception as e:
            self.showMessage("Некорректный файл таблицы Лапласа")
            return False

        try:
            with open('Student.txt', 'r') as f:
                lines = f.readlines()
        except IOError as e:
            self.showMessage("Файл с таблицей Стьюдента не найден")
            return False
        try:
            self.student_table = []
            for i in lines:
                self.student_table.append([*(map(float, i.strip('\n').split(' ')))])
        except Exception as e:
            self.showMessage("Некорректный файл таблицы Стьюдента")
            return False

        try:
            with open('hi.txt', 'r') as f:
                lines = f.readlines()
        except IOError as e:
            self.showMessage("Файл с таблицей хи^2 не найден")
            return False
        try:
            self.hi2_table = []
            for i in lines:
                self.hi2_table.append([*(map(float, i.strip('\n').split(' ')))])
        except Exception as e:
            self.showMessage("Некорректный файл таблицы хи^2")
            return False

        return True

    # расчет 1-3 задания 1 части
    def pushButtonPart1(self):
        if not self.checkInputFirst():
            return

        # 1
        self.distr_values = pf.getDistrValues(0, 3, self.num_of_val, self.param_A, self.param_B)
        self.puass_values = pf.getPuassValues(0, 20, self.num_of_val, self.param_lambda)

        self.fillTableFirst(self.p1_t, self.distr_values)
        self.fillTableFirst(self.p2_t, self.puass_values)
        #2
        self.sampAverDistr = pf.getSampleAverage(self.distr_values)
        self.sampAverPuass = pf.getSampleAverage(self.puass_values)
        self.sampVarDisrt = pf.getSampleVariance(self.distr_values, self.sampAverDistr)
        self.sampVarPuass = pf.getSampleVariance(self.puass_values, self.sampAverPuass)

        self.mx_p1_l.setText(f'{self.sampAverDistr:.4f}')
        self.mx_p2_l.setText(f'{self.sampAverPuass:.4f}')
        self.dx_p1_l.setText(f'{self.sampVarDisrt:.4f}')
        self.dx_p2_l.setText(f'{self.sampVarPuass:.4f}')

        #3
        self.mx_p2_l_2.setText(str(f'{self.sampAverPuass:.4f}'))

        self.valuesReady(True)

    # расчет 4-5 задания 1 части
    def pushButtonPart1_45(self):
        try:
            p_prob = float(self.param_p_l.text())
            if not (0 <= p_prob < 1):
                raise Exception
        except Exception:
            self.showMessage("Заданная вероятность - число от 0 до 1")
            return

        # №4

        ProbPuass = 0
        l = 0
        while ProbPuass < p_prob:
            a = round(self.param_lambda - (l / 9))
            b = round(self.param_lambda + ((5 * l) / 2))
            ProbPuass = 0
            if a < 0:
                a = 0
            for i in range(a, b):
                ProbPuass += pf.getPuass(self.param_lambda, i)
            l += 0.1

        self.l4_p2.setText(f'{l:.4f}')
        self.int4_left_p2.setText(f'{self.sampAverPuass - (l / 9):.4f}')
        self.int4_right_p2.setText(f'{self.sampAverPuass + ((5 * l) / 2):.4f}')

        # №5
        leftDistrD, rightDistrD, lDistrD = pf.getProbInterval(p_prob, self.sampAverDistr, self.distr_values)
        leftPuassD, rightPuassD, lPuassD = pf.getProbInterval(p_prob, self.sampAverPuass, self.puass_values)

        self.l5_p1.setText(f'{lDistrD:.4f}')
        self.int5_left_p1.setText(f'{leftDistrD:.4f}')
        self.int5_right_p1.setText(f'{rightDistrD:.4f}')

        self.l5_p2.setText(f'{lPuassD:.4f}')
        self.int5_left_p2.setText(f'{leftPuassD:.4f}')
        self.int5_right_p2.setText(f'{rightPuassD:.4f}')

    # расчет 1-6 задания 2 части
    def pushButtonPart2(self):
        if not self.checkInputSecond():
            return
        # №1
        self.varInt, self.varValues = pf.getVarInter(self.data_values, self.num_of_int)
        self.interval_t.clear()
        self.interval_t.setColumnCount(2)
        self.interval_t.setRowCount(0)
        self.interval_t.setHorizontalHeaderLabels(["Интервал", "Количество элементов"])
        for i in range(0, len(self.varInt)):
            self.interval_t.insertRow(self.interval_t.rowCount())
            row = self.interval_t.rowCount() - 1
            self.interval_t.setItem(row, 0, QTableWidgetItem(f'[{self.varInt[i][0]:.1f}, {self.varInt[i][1]:.1f})'))
            self.interval_t.setItem(row, 1, QTableWidgetItem(f'{len(self.varValues[i])}'))

        # №3
        sampAverData = pf.getSampleAverage(self.data_values)
        self.param3_l.setText(f'{sampAverData:.4f}')

        # №4
        sampAverVar = pf.getSampleVariance(self.data_values, sampAverData)
        gamma = self.param_gamma / 2
        t = min(self.laplas_table, key=lambda a: abs(a[1] - gamma))[0]
        eps = t * np.sqrt(sampAverVar)
        average_val = sum(self.data_values) / len(self.data_values)

        self.eps4_l.setText(f'{eps:.4f}')
        self.int4_left_2.setText(f'{average_val-eps:.4f}')
        self.int4_right_2.setText(f'{average_val+eps:.4f}')

        # №5
        dataLen = len(self.data_values)
        gamma = float(self.p5_cb.currentText()) / 2
        if (dataLen - 1) > 120:
            k = 120
        else:
            k = dataLen - 1

        t = self.student_table[k - 1][self.student_prob[gamma]]
        eps = (t * np.sqrt(sampAverVar)) / np.sqrt(k)

        self.eps5_l.setText(f'{eps:.4f}')
        self.int5_left_2.setText(f'{average_val - eps:.4f}')
        self.int5_right_2.setText(f'{average_val + eps:.4f}')

        # №6
        gamma = float(self.p6_cb.currentText())
        intervals = copy.deepcopy(self.varInt)
        intVal = [*map(lambda x: len(x), self.varValues)]
        inter_check = False

        while not inter_check:
            inter_check = True
            for i in range(0, len(intVal)):
                if intVal[i] < 5:
                    inter_check = False
                    if len(intVal) <= 2:
                        self.showMessage("Неудачный расчет задания 6. Введите меньше интервалов")
                        inter_check = True
                        break
                    if i == 0:
                        intervals[0][1] = intervals[1][1]
                        intVal[0] += intVal[1]
                        intervals.pop(1)
                        intVal.pop(1)
                    else:
                        intervals[i][0] = intervals[i - 1][0]
                        intVal[i] += intVal[i - 1]
                        intervals.pop(i - 1)
                        intVal.pop(i - 1)
                    break

        average = []
        for i in range(0, len(intervals)):
            average.append((intervals[i][0] + intervals[i][1]) / 2)

        a = 0
        for i in range(0, len(average)):
            a += (average[i] * (intVal[i]))
        a = a / dataLen

        D = 0
        for i in range(0, len(average)):
            D += ((average[i] - a) ** 2) * intVal[i]
        D = D / dataLen

        sigma = np.sqrt(D)

        p = []
        F2 = (intervals[0][1] - a) / sigma
        t2 = min(self.laplas_table, key=lambda a: abs(a[0] - abs(F2)))[1]
        if F2 < 0:
            t2 = -t2
        p.append((t2 + 1 / 2) * dataLen)
        for i in range(1, len(intervals) - 1):
            F1 = (intervals[i][0] - a) / sigma
            F2 = (intervals[i][1] - a) / sigma
            t1 = min(self.laplas_table, key=lambda a: abs(a[0] - abs(F1)))[1]
            t2 = min(self.laplas_table, key=lambda a: abs(a[0] - abs(F2)))[1]
            if F1 < 0:
                t1 = -t1
            if F2 < 0:
                t2 = -t2
            p.append((t2 - t1) * dataLen)

        F1 = (intervals[-1][0] - a) / sigma
        t1 = min(self.laplas_table, key=lambda a: abs(a[0] - abs(F1)))[1]
        if F1 < 0:
            t1 = -t1
        p.append((1 / 2 - t1) * dataLen)

        hi2 = 0
        for i in range(0, len(intervals)):
            hi2 += (intVal[i] ** 2) / p[i]
        hi2 -= len(intervals)

        k = len(intVal) - 2 - 1
        hi2_t = self.hi2_table[k - 1][self.hi2_prob[gamma]]

        self.hi_r_l.setText(f'{hi2:.4f}')
        self.hi_t_l.setText(f'{hi2_t:.4f}')

        if hi2 <= hi2_t:
            self.label_h0.setText('принимается')
        else:
            self.label_h0.setText('не принимается')

        self.calcFuture()
        self.valuesReadySecond(True)

    # расчет 7 задания 2 части
    def calcFuture(self):

        average = []
        dataLen = len(self.data_values)
        for i in range(0,4):
            average.append(sum(self.data_values[dataLen-i-4:dataLen-i])/4)
        choosen_value = 1/24*(55*average[0]-59*average[1]+37*average[2]-9*average[3])
        self.val7_l.setText(f'{int(choosen_value)}')

    # вывод графиак второй части
    def showPlot(self):
        plt.close()
        plt.figure()
        prob = []
        count= len(self.data_values)
        for i in self.varValues:
            prob.append(len(i) / count)

        y_ax = []
        for i in range(0, len(prob)):
            y_ax.append(sum(prob[0:i]))
        x_ax = []
        for i in self.varInt:
            x_ax.append([i[0], i[1]])
        for i in range(0, len(self.varInt)):
            plt.plot(x_ax[i], [y_ax[i]] * 2, 'r')
        plt.grid(True)
        plt.show()

