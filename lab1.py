"""Лабораторная работа №1. Метод прогонки
Выполнил: Калачиков Алексей Юрьевич
Группа: 4-ИАИТ-10
Вариант: 7"""

import math
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

print(
    "Краевая задача вида: \n -p(x) * y'' + q(x) * y' + r(x) * y = f(x), a<=x<=b, "
    "\n alpha_1 * y(a) - alpha_2 * y'(a) = alpha, "
    "\n beta_1 * y(b) + beta_2 * y'(b) = beta. \nЧисленное решение представляет порядок точности O(h).")
# Исходные данные для задачи
alpha_1 = 0.0
alpha_2 = -1.0
beta_1 = 1.0
beta_2 = 0.0
alpha = 32.5
beta = 82.1
n = 10
a = 2.0
b = 3.0
h = (b - a) / n
x_i_extended = [0.0] * (n + 2)
for i in range(0, n + 2):
    x_i_extended[i] = round(a - h / 2 + i * h, 5)
print(f'\nРазбиение: {x_i_extended}')
p = -1
q = 1 / x_i_extended[i]
r = -1
p_i = [p] * (n + 2)
r_i = [r] * (n + 2)
q_i = [q] * (n + 2)
A = [0.0] * (n + 2)
A[n + 1] = beta_1 / 2 - beta_2 / h
for i in range(1, n + 1):
    A[i] = -p_i[i] / (h ** 2) - q_i[i] / (2 * h)
B = [0.0] * (n + 2)
B[0] = -alpha_1 / 2 - alpha_2 / h
B[n + 1] = -(beta_2 / h + beta_1 / 2)
for i in range(1, n + 1):
    B[i] = -2 * p_i[i] / (h ** 2) - r_i[i]
C = [0.0] * (n + 2)
C[0] = alpha_1 / 2 - alpha_2 / h
C[n + 1] = 0
for i in range(1, n + 1):
    C[i] = -p_i[i] / (h ** 2) + q_i[i] / (2 * h)
G = [0.0] * (n + 2)
G[0] = alpha
G[n + 1] = beta
for i in range(1, n + 1):
    G[i] = 16 * x_i_extended[i] ** 2 - x_i_extended[i] ** 4 - math.log(x_i_extended[i])
s = [0.0] * (n + 2)
s[0] = C[0] / B[0]
s[n + 1] = 0
for i in range(1, n + 1):
    s[i] = C[i] / (B[i] - A[i] * s[i - 1])
t = [0.0] * (n + 2)
t[0] = -G[0] / B[0]
for i in range(1, n + 2):
    t[i] = (A[i] * t[i - 1] - G[i]) / (B[i] - A[i] * s[i - 1])
y_approximate = [0.0] * (n + 2)
y_approximate[n + 1] = t[n + 1]
for i in range(n, -1, -1):
    y_approximate[i] = s[i] * y_approximate[i + 1] + t[i]
y = [x ** 4 * math.log(x) for x in x_i_extended]
y_pow_2 = [y_i ** 2 for y_i in y]
Q_ocm = [(y_i - y_i_app) ** 2 for y_i, y_i_app in zip(y, y_approximate)]
numerator = sum(Q_ocm)
denominator = sum(y_pow_2)
error = numerator / denominator * 100
print(f'Погрешность = {round(error, 5)}\n')


def approximation():
    """Аппроксимация первого порядка, метод прогонки"""
    my_table = PrettyTable()
    my_table_names = ['i', "x_i", "A_i", "B_i", "C_i", "G_i", "s_i", "t_i", "y_i_approximate", "y_i"]
    table_helper = [
        [i for i in range(0, n + 2)],
        [round(item, 5) for item in x_i_extended],
        [round(item, 5) for item in A],
        [round(item, 5) for item in B],
        [round(item, 5) for item in C],
        [round(item, 5) for item in G],
        [round(item, 5) for item in s],
        [round(item, 5) for item in t],
        [round(item, 5) for item in y_approximate],
        [round(item, 5) for item in y]
    ]
    for column, name_column in zip(table_helper, my_table_names):
        my_table.add_column(name_column, column)
    print(my_table)


def builder():
    """Построение графика"""
    solution = lambda z: z ** 4 * np.log(z)
    domain = np.linspace(a - h / 2, b + h / 2, 100)
    plt.plot(domain, solution(z=domain))
    plt.scatter(x_i_extended, y_approximate)
    plt.show()


if __name__ == '__main__':
    approximation()
    builder()
