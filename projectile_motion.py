"""
Znajdź trajektorię piłki rzuconej ukośnie do powierzchni Ziemi
  1. zaniedbując opory powietrza,
  2. uwzględniając opory powietrza.

W drugim przypadku załóż, że siła oporu powietrza jest postaci
$$\vec{F}=-\frac{1}{2}c_w\rho A\vec{v}|\vec{v}|\,,$$
gdzie
- $c_w\approx 0.35$ - współczynnik oporu powietrza,
- $\rho\approx 1.2\,$kg/m$^3$ - gęstosć powietrza,
- $A[$m$^2]$ - pole przekroju poprzecznego piłki
- $\vec{v}$ - prędkość piłki

Obliczenia przeprowadź dla różnych wielkości piłki, prędkości początkowych i kątów rzutu. Wyniki przedstaw na wykresie
"""

import numpy as np
import matplotlib.pyplot as plt

def f3b(v0, theta, A=0.01, drag = False):
    g = 9.81
    m = 1
    t_max = 2 * v0 * np.sin(theta) / g #?
    t = np.linspace(0, t_max, 100)
    if drag: # z oporem
        x = m * np.log(0.21 * A * t * v0 * np.cos(theta) + m) / (0.21 * A)
        y = np.zeros(len(t))
        t_break = 0
        diff = 0
        for i in range(len(y)):
            vy = v0*np.sin(theta) - g * t[i]
            if vy > 0:
                y[i] = (m / (0.42 * A) ) * np.log( (m * g + 0.21 * A * (vy + g * t[i])**2) / (m * g + 0.21 * A * vy**2) )
            else:
                vy_br = v0*np.sin(theta) - g * t[i-1]
                t_break = i
                diff = np.abs( (m / (0.42 * A) ) * np.log( (m * g - 0.21 * A * (vy + g * t[i])**2) / (m * g + 0.21 * A * vy**2) ) ) + y[t_break-1]
                break
        for i in range(t_break, len(y)):
            vy = v0*np.sin(theta) - g * t[i]
            y[i] = (m / (0.42 * A) ) * np.log( (m * g - 0.21 * A * (vy + g * t[i])**2) / (m * g + 0.21 * A * vy**2) ) + diff
    else: #bez oporu:
        x = v0 * np.cos(theta) * t
        y = v0 * np.sin(theta) * t - 0.5 * g * t**2

    return x, y


# opór vs brak oporu:
v0s = [5]
ths = [np.pi/4]
As = [1]

plt.figure(figsize=(12, 5))
for v0 in v0s:
    for th in ths:
        for A in As:
            x, y = f3b(v0, th, A)
            plt.plot(x, y, label=f'v0={v0}, th={np.degrees(th):.0f}, A={A}, drag off')
            x, y = f3b(v0, th, A, True)
            plt.plot(x, y, label=f'v0={v0}, th={np.degrees(th):.0f}, A={A}, drag on')

plt.legend()
plt.title("Opór vs brak oporu")
plt.grid(True)
plt.show()


# zmiana kata:
v0s = [5]
ths = [np.pi/4, np.pi/5, np.pi/6]
As = [0.5]

plt.figure(figsize=(12, 5))
for v0 in v0s:
    for th in ths:
        for A in As:
            x, y = f3b(v0, th, A, True)
            plt.plot(x, y, label=f'v0={v0}, th={np.degrees(th):.0f}, A={A}, drag on')

plt.legend()
plt.grid(True)
plt.title("Zmiana kąta")
plt.show()


# zmiana prędkości:
v0s = [5,6,7]
ths = [np.pi/4]
As = [0.5]

plt.figure(figsize=(12, 5))
for v0 in v0s:
    for th in ths:
        for A in As:
            x, y = f3b(v0, th, A, True)
            plt.plot(x, y, label=f'v0={v0}, th={np.degrees(th):.0f}, A={A}, drag on')

plt.legend()
plt.grid(True)
plt.title("Zmiana prędkości")
plt.show()


# zmiana pola powierzchni:
v0s = [5]
ths = [np.pi/4]
As = [0.5,1,1.5]

plt.figure(figsize=(12, 5))
for v0 in v0s:
    for th in ths:
        for A in As:
            x, y = f3b(v0, th, A, True)
            plt.plot(x, y, label=f'v0={v0}, th={np.degrees(th):.0f}, A={A}, drag on')

plt.legend()
plt.grid(True)
plt.title("Zmiana pola powierzchni")
plt.show()
