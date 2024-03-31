"""
Rozwiąż metodą strzałów równanie różniczkowe
$$\frac{d^2y}{dt^2}+\sin(y)+1=0$$\
z następującymi warunkami brzegowymi $y(0) = 0$ i $y(\pi)=0$.
"""

"""
Rozpisuję na układ dwóch równań liniowych:
$$\frac{dy}{dt}=v$$
$$\frac{dv}{dt}=-\sin(y)-1$$
Z kilku wstępnych prób wynika że $s$ z równania $v(0) = s$ leży w przedziale od 2 do 3.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def system(t, y):
    dydt = y[1]
    dvdt = -np.sin(y[0]) - 1
    return [dydt, dvdt]

guesses = np.linspace(2, 3, 1000)
vals = []

for s in guesses:
    y0 = [0, s]  # y(0) = 0, v(0) = s
    t = (0, 4)

    sol = solve_ivp(system, t, y0, dense_output=True)
    vals.append([s, np.abs(sol.sol(np.pi)[0])])  # sol.sol(np.pi)[0] czyli wartość y(pi)

bestS = min(vals, key=lambda x: x[1])  # szukamy które s osiągnęło y(pi) najbliższe do 0
print(f"First try:\ns = {bestS[0]:.8f}, y(pi) = {bestS[1]:.8f}")


# Aby uzyskać dokładniejszy wynik, powtarzam procedurę dla zakresu od 2.80 do 2.85.


guesses = np.linspace(2.8, 2.85, 1000)
vals = []
t = (0, 4)

for s in guesses:
    y0 = [0, s]  # y(0) = 0, v(0) = s

    sol = solve_ivp(system, t, y0, dense_output=True)
    vals.append([s, np.abs(sol.sol(np.pi)[0])])  # sol.sol(np.pi)[0] czyli wartość y(pi)

bestS = min(vals, key=lambda x: x[1])  # szukamy które s osiągnęło y(pi) najbliższe do 0
print(f"\nSecond try:\ns = {bestS[0]:.8f}, y(pi) = {bestS[1]:.8f}")

# Wyrysowanie najlepszego rozwiązania
besty0 = [0, bestS[0]]
bestSol = solve_ivp(system, t, besty0, dense_output=True)
t1 = np.linspace(0, 4, 100)
y = bestSol.sol(t1)

plt.plot(t1, y[0], label=f"krzywa dla y(pi) = {bestS[1]:.5f}")
plt.scatter(np.pi, 0, color='red', label="y(pi)=0")
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.title("Rozwiązanie równania")
plt.show()
