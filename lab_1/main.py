import requests
import numpy as np
import matplotlib.pyplot as plt

url = f"https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"

response = requests.get(url)
all_results = response.json()["results"]



def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlamb = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlamb / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


full_dist = [0]
full_elev = [p["elevation"] for p in all_results]
for i in range(1, len(all_results)):
    d = haversine(all_results[i - 1]["latitude"], all_results[i - 1]["longitude"],
                  all_results[i]["latitude"], all_results[i]["longitude"])
    full_dist.append(full_dist[-1] + d)

X_all, Y_all = np.array(full_dist), np.array(full_elev)


def get_spline_coeffs(x, y):
    n = len(x) - 1
    h = np.diff(x)
    al, be, ga, de = np.zeros(n + 1), np.ones(n + 1), np.zeros(n + 1), np.zeros(n + 1)
    for i in range(1, n):
        al[i], be[i], ga[i] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
        de[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    A, B = np.zeros(n + 1), np.zeros(n + 1)
    for i in range(1, n + 1):
        m = al[i] * A[i - 1] + be[i]
        A[i], B[i] = -ga[i] / m, (de[i] - al[i] * B[i - 1]) / m
    c = np.zeros(n + 1)
    c[n] = B[n]
    for i in range(n - 1, -1, -1):
        c[i] = A[i] * c[i + 1] + B[i]
    a = y[:-1]
    b = np.array([(y[i + 1] - y[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3 for i in range(n)])
    d = np.array([(c[i + 1] - c[i]) / (3 * h[i]) for i in range(n)])
    return a, b, c[:-1], d


def evaluate_spline(x_target, x_nodes, a, b, c, d):
    y_target = []
    for val in x_target:
        idx = np.searchsorted(x_nodes, val) - 1
        idx = max(0, min(idx, len(a) - 1))
        dx = val - x_nodes[idx]
        y_target.append(a[idx] + b[idx] * dx + c[idx] * dx ** 2 + d[idx] * dx ** 3)
    return np.array(y_target)


node_counts = [10, 15, 20]
colors = ['blue', 'orange', 'purple']
x_fine = np.linspace(X_all[0], X_all[-1], 300)

a_ref, b_ref, c_ref, d_ref = get_spline_coeffs(X_all, Y_all)
y_ref = evaluate_spline(x_fine, X_all, a_ref, b_ref, c_ref, d_ref)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

ax1.plot(x_fine, y_ref, color='green', linewidth=2, linestyle='--', label='Еталон (всі точки)')
ax1.scatter(X_all, Y_all, color='black', s=20, alpha=0.5, label='Всі вузли GPS')

for count, color in zip(node_counts, colors):
    indices = np.linspace(0, len(X_all) - 1, count, dtype=int)
    X_sub, Y_sub = X_all[indices], Y_all[indices]

    a, b, c, d = get_spline_coeffs(X_sub, Y_sub)
    y_approx = evaluate_spline(x_fine, X_sub, a, b, c, d)

    ax1.plot(x_fine, y_approx, color=color, label=f'Сплайн ({count} вузлів)')
    ax1.scatter(X_sub, Y_sub, color=color, s=40, edgecolors='black')

ax1.set_title("Порівняння інтерполяції кубічними сплайнами")
ax1.set_ylabel("Висота (м)")
ax1.legend()
ax1.grid(True)

for count, color in zip(node_counts, colors):
    indices = np.linspace(0, len(X_all) - 1, count, dtype=int)
    X_sub, Y_sub = X_all[indices], Y_all[indices]

    a, b, c, d = get_spline_coeffs(X_sub, Y_sub)
    y_approx = evaluate_spline(x_fine, X_sub, a, b, c, d)

    error = np.abs(y_ref - y_approx)
    ax2.plot(x_fine, error, color=color, label=f'Похибка ({count} вузлів)')

ax2.set_title("Графік абсолютної похибки: |y_ref - y_approx|")
ax2.set_xlabel("Відстань (м)")
ax2.set_ylabel("Похибка (м)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()