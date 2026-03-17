import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

csv_filename = "data.csv"
if not os.path.exists(csv_filename):
    with open(csv_filename, "w", encoding="utf-8") as f:
        f.write("Objects,FPS\n100,120\n200,110\n400,90\n800,65\n1600,40\n")

df = pd.read_csv(csv_filename)
x_nodes = df['Objects'].values.astype(float)
y_nodes = df['FPS'].values.astype(float)

def omega_k(x, nodes, k):
    """Знаходження значення omega_k(x) = П(x - x_i)"""
    res = 1.0
    for i in range(k):
        res *= (x - nodes[i])
    return res


def divided_differences_table(x, y):
    """Обчислення розділених різниць f(x0,...,xk)"""
    n = len(y)
    table = np.zeros([n, n])
    table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x[i + j] - x[i])
    return table


def Newton_Nn(x, nodes, diff_table):
    """Знаходження значення багаточлена Ньютона N_n(x)"""
    n = diff_table.shape[1]
    res = diff_table[0, 0]
    for k in range(1, n):
        res += diff_table[0, k] * omega_k(x, nodes, k)
    return res


def true_f(x):
    """Гіпотетична модель рушія для розрахунку похибок"""
    return 130 * np.exp(-0.00075 * x)

def finite_differences_table(y):
    """Таблиця скінченних різниць"""
    n = len(y)
    table = np.zeros([n, n])
    table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = table[i + 1][j - 1] - table[i][j - 1]
    return table


def factorial_poly_predict(x_target, x_nodes, y_nodes):
    """Прогноз факторіальним многочленом.
    z = log2(x / 100), щоб отримати сітку 0, 1, 2, 3, 4 (h=1)."""
    z_nodes = np.log2(x_nodes / 100.0)
    z_target = np.log2(x_target / 100.0)

    fin_table = finite_differences_table(y_nodes)
    n = len(y_nodes)

    res = fin_table[0, 0]
    t_k = 1.0
    for k in range(1, n):
        t_k *= (z_target - k + 1)
        res += (fin_table[0, k] * t_k) / math.factorial(k)
    return res


diff_table_base = divided_differences_table(x_nodes, y_nodes)

fps_1000_newton = Newton_Nn(1000, x_nodes, diff_table_base)
fps_1000_fact = factorial_poly_predict(1000, x_nodes, y_nodes)

x_range = np.linspace(100, 1600, 5000)
y_interp_base = [Newton_Nn(val, x_nodes, diff_table_base) for val in x_range]
limit_60fps = x_range[np.argmin(np.abs(np.array(y_interp_base) - 60))]

print("\n" + "=" * 50)
print("ОСНОВНІ РЕЗУЛЬТАТИ (Варіант 5):")
print("=" * 50)
print(f"Прогноз FPS (1000 об'єктів, Ньютон): {fps_1000_newton:.2f}")
print(f"Прогноз FPS (1000 об'єктів, Факторіальний): {fps_1000_fact:.2f}")
print(f"Межа для 60 FPS: ~{int(limit_60fps)} об'єктів")

fig, ax = plt.subplots(1, 2, figsize=(15, 6))

ax[0].plot(x_range, y_interp_base, 'b-', label='Інтерполяція (Ньютон)')
ax[0].plot(x_nodes, y_nodes, 'ro', markersize=8, label='Дані з CSV')
ax[0].plot(1000, fps_1000_newton, 'c*', markersize=10, label=f'Прогноз (FPS={fps_1000_newton:.1f})')
ax[0].axhline(60, color='gray', linestyle='--', label='60 FPS')
ax[0].set_title("Прогноз продуктивності: FPS = f(Objects)")
ax[0].set_xlabel("Кількість об'єктів")
ax[0].set_ylabel("FPS")
ax[0].legend()
ax[0].grid(True)

w_values = [omega_k(x, x_nodes, len(x_nodes)) for x in x_range]
ax[1].plot(x_range, w_values, 'g-')
ax[1].axhline(0, color='black', linewidth=1)
ax[1].set_title(r"Графік функції $\omega_n(x)$")
ax[1].set_xlabel("x")
ax[1].grid(True)

plt.tight_layout()
plt.show()

print("\n" + "=" * 75)
print("ДОСЛІДЖЕННЯ: Фіксований крок (h=50), змінний інтервал")
print("-" * 75)

plt.figure(figsize=(10, 6))
h_fixed = 50
a_start = 100

for n_nodes in [5, 10, 20]:
    b_end = a_start + h_fixed * (n_nodes - 1)

    x_test_nodes = np.linspace(a_start, b_end, n_nodes)
    y_test_nodes = true_f(x_test_nodes)
    table_test = divided_differences_table(x_test_nodes, y_test_nodes)

    x_dense_test = np.linspace(a_start, b_end, 500)
    errors = [abs(true_f(val) - Newton_Nn(val, x_test_nodes, table_test)) for val in x_dense_test]

    errors = np.where(np.array(errors) < 1e-16, 1e-16, errors)

    plt.plot(x_dense_test, errors, label=f'n={n_nodes} (Інтервал [{a_start}, {b_end}])')
    print(f"n = {n_nodes:<2} | Інтервал: [{a_start}, {b_end:<4}] | Макс. похибка: {max(errors):.2e}")

plt.yscale('log')
plt.title("Вплив кількості вузлів при фіксованому кроці (h=50)")
plt.xlabel("x")
plt.ylabel("Абсолютна похибка")
plt.legend()
plt.grid(True)
plt.show()

def runge_function(x): return 1 / (1 + 0.0005 * x ** 2)


plt.figure(figsize=(10, 6))
x_runge_dense = np.linspace(-200, 200, 1000)

plt.plot(x_runge_dense, runge_function(x_runge_dense), 'k--', linewidth=2, label="Функція Рунге")

for n_nodes in [5, 10, 20]:
    x_r = np.linspace(-200, 200, n_nodes)
    y_r = runge_function(x_r)
    t_r = divided_differences_table(x_r, y_r)
    y_i = [Newton_Nn(val, x_r, t_r) for val in x_runge_dense]
    plt.plot(x_runge_dense, y_i, label=f"Ньютон (n={n_nodes})")

plt.ylim(-0.5, 1.5)
plt.title("Ефект Рунге: поліноміальна інтерполяція на рівномірній сітці")
plt.legend()
plt.grid(True)
plt.show()