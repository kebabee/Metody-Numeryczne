"""
Równanie ruchu wahadła matematycznego z tłumieniem oraz okresową siłą wymuszającą można przedstawić w postaci

$$\frac{d^2\theta}{d\tau^2}+\frac{1}{Q}\frac{d\theta}{d\tau}+\sin\theta=A\cos(\tau\bar{\omega})$$

gdzie $\omega_0=\sqrt{\frac{g}{l}}$, $\tau=\omega_0t$ i $\bar{\omega}=\frac{\omega}{\omega_0}\,$.

Rozwiąż to równanie funkcją *scipy.integrate.solv_ivp()* dla:

- $Q=2$, $\bar{\omega}=2/3$, $A=0.5$, $\dot{\theta}_0=0$, $\theta_0=0.01$;
- $Q=2$, $\bar{\omega}=2/3$, $A=0.5$, $\dot{\theta}_0=0$, $\theta_0=0.3$;
- $Q=2$, $\bar{\omega}=2/3$, $A=1.35$, $\dot{\theta}_0=0$, $\theta_0=0.3$;

Przedstaw na wykresach 
- zależności $\theta(t)$
- trajektorie w przestrzeni fazowej $(\dot{\theta},\theta)$
"""

# Dla uproszczenia zakładam długośc wahadła $l = g$, wówczas $\omega_0$ = 1, z tego wynika $\omega = \bar{\omega} * \omega_0 = \bar{\omega}$ oraz $\tau = t$.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def f1(t, y, Q, w, A):
    theta, theta_dot = y
    dydt = theta_dot
    eq = -1/Q * theta_dot - np.sin(theta) + A * np.cos(w * t)
    return [dydt, eq]

cond = [[0.01, 0], [0.3, 0], [0.3, 0], [0.3, 0], [0.3, 0]]
Q = [2, 2, 4, 2, 2]
w = [2/3, 2/3, 2/3, 2/3, 2/3]
A = [0.5, 0.5, 0.5, 1.35, 2.35]
t = [0, 100]

for i in range(len(cond)):
    sol = solve_ivp( f1, t, cond[i], args=[Q[i], w[i], A[i]], max_step=0.1)

    # theta od t
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(sol.t, sol.y[0])
    plt.title(f'theta(t) dla Q={Q[i]}, w={w[i]:.3f}, A={A[i]}')
    plt.xlabel('t')
    plt.ylabel('theta(t)')

    # przestrzeń fazowa
    plt.subplot(1, 2, 2)
    plt.plot(sol.y[1], sol.y[0])
    plt.title(f'trajektoria dla Q={Q[i]}, w={w[i]:.3f}, A={A[i]}')
    plt.xlabel('thetaDot')
    plt.ylabel('theta')

    plt.show()
