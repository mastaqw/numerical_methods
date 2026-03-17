import numpy as np
import matplotlib.pyplot as plt
import csv
import os

filename = os.path.join(os.path.dirname(__file__), 'data.csv')


def load_data(filename):
    months = []
    temps = []
    try:
        with open(filename, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                months.append(float(row['Month']))
                temps.append(float(row['Temp']))
    except FileNotFoundError:
        print(f"Файл {filename} не знайдено.")

    return np.array(months, dtype=np.float64), np.array(temps, dtype=np.float64)


x, y = load_data('data.csv')


# Функції Методу найменших квадратів (МНК)
def form_matrix(x, m):
    """ Формування матриці системи """
    A = np.zeros((m + 1, m + 1), dtype=np.float64)
    for i in range(m + 1):
        for j in range(m + 1):
            A[i, j] = np.sum(x ** (i + j))
    return A


def form_vector(x, y, m):
    """ Формування вектора вільних членів """
    b = np.zeros(m + 1, dtype=np.float64)
    for i in range(m + 1):
        b[i] = np.sum(y * (x ** i))
    return b


def gauss_solve(A_in, b_in):
    """ Метод Гауса з вибором головного елемента по стовпцю """
    A = A_in.copy()
    b = b_in.copy()
    n = len(b)

    # Прямий хід
    for k in range(n - 1):
        max_row = np.argmax(np.abs(A[k:n, k])) + k
        A[[k, max_row]] = A[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]

        for i in range(k + 1, n):
            if A[k, k] == 0: continue
            factor = A[i, k] / A[k, k]
            A[i, k:] = A[i, k:] - factor * A[k, k:]
            b[i] = b[i] - factor * b[k]

    # Зворотній хід
    x_sol = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        if A[i, i] != 0:
            x_sol[i] = (b[i] - np.sum(A[i, i + 1:] * x_sol[i + 1:])) / A[i, i]
    return x_sol


def polynomial(x, coef):
    """ Обчислення значення полінома """
    y_poly = np.zeros_like(x, dtype=np.float64)
    for i in range(len(coef)):
        y_poly += coef[i] * (x ** i)
    return y_poly


def variance(y_true, y_approx):
    """ Дисперсія"""
    return np.sum((y_true - y_approx) ** 2) / (len(y_true) + 1)

max_degree = 10
variances = []
optimal_m = 3

print("--- Дисперсії для різних степенів ---")
for m in range(1, max_degree + 1):
    A_m = form_matrix(x, m)
    b_vec_m = form_vector(x, y, m)
    coef_m = gauss_solve(A_m, b_vec_m)
    y_approx_m = polynomial(x, coef_m)
    var = variance(y, y_approx_m)
    variances.append(var)
    print(f"Ступінь m={m}: дисперсія = {var:.4f}")

coef_opt = gauss_solve(form_matrix(x, optimal_m), form_vector(x, y, optimal_m))
y_approx_opt = polynomial(x, coef_opt)

# Прогноз
x_future = np.array([25, 26, 27], dtype=np.float64)
y_future_opt = polynomial(x_future, coef_opt)

m_high = 10
coef_high = gauss_solve(form_matrix(x, m_high), form_vector(x, y, m_high))
y_future_high = polynomial(x_future, coef_high)

print(f"\n=> Прогноз (m=3) на 25, 26, 27 місяці: {np.round(y_future_opt, 2)}")
print(f"=> Прогноз (m=10) на 25, 26, 27 місяці: {np.round(y_future_high, 2)}")

plt.figure(figsize=(16, 10))

# Прогноз на три місяці (m=3)
plt.subplot(2, 2, 1)
plt.plot(x, y, 'o', label='Фактичні дані', color='blue')
x_dense = np.linspace(min(x), max(x), 100)
plt.plot(x_dense, polynomial(x_dense, coef_opt), '-', label=f'Апроксимація (m={optimal_m})', color='green')
plt.plot(x_future, y_future_opt, 'X', color='red', markersize=8, label='Прогноз')
plt.title('Прогноз на три місяці')
plt.grid(True)
plt.legend()

# Залежність дисперсії від степеня m
plt.subplot(2, 2, 2)
m_values = list(range(1, max_degree + 1))
plt.plot(m_values, variances, marker='s', color='blue', linestyle='-')
plt.title('Залежність дисперсії від степені m')
plt.xlabel('Ступінь многочлена m')
plt.ylabel('Дисперсія')
plt.xticks(m_values)
plt.grid(True)

# Графік похибки апроксимацій
plt.subplot(2, 2, 3)
for m in range(1, max_degree + 1):
    color_val = (m - 1) / (max_degree - 1)
    line_color = (1 - color_val, color_val, 0)

    A_temp = form_matrix(x, m)
    b_temp = form_vector(x, y, m)
    coef_temp = gauss_solve(A_temp, b_temp)
    y_approx_temp = polynomial(x, coef_temp)
    error_temp = np.abs(y - y_approx_temp)
    plt.plot(x, error_temp, '-o', label=f'm={m}', color=line_color, alpha=0.8, markersize=4)

plt.title('Графік похибки апроксимацій')
plt.xlabel('Вузли (Місяці)')
plt.ylabel('Похибка')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

# Апроксимація 10-м степенем
plt.subplot(2, 2, 4)
x_dense_high = np.linspace(min(x), max(x), 200)
y_high_dense = polynomial(x_dense_high, coef_high)

plt.plot(x, y, 'o', label='Фактичні дані', color='blue', alpha=0.5)
plt.plot(x_dense_high, y_high_dense, '-', label=f'Апроксимація (m={m_high})', color='purple')
plt.plot(x_future, y_future_high, 'X', color='red', markersize=8, label='Прогноз (m=10)')
plt.title(f'Апроксимація та прогноз 10-м степенем')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()