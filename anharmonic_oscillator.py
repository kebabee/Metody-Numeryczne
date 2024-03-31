"""
Znajdź cztery najmniejsze wartości własne kwantowego oscylatora anharmonicznego, którego równanie Schrodingera ma postać
$$-\frac{1}{2}\frac{d^2\psi}{dx^2}+\left(\frac{x^2}{2}+\lambda x^4\right)\psi(x)=E\psi(x)$$
z następującymi warunkami brzegowymi $y(0) = 0$ i $y(\pi)=0$. 
"""

import numpy as np
import matplotlib.pyplot as plt

def H1(n):
    matrix = np.zeros((n,n))
    for i in range(n):
        matrix[i][i] = d1[i+1]
        if i + 1 < n:
            matrix[i + 1][i] = g
            matrix[i][i + 1] = g
    return matrix

# Stałe
a = 4.6
lam = 0.2
m = 1000
h = 2 * a / m
g = -1 / (2 * h**2)

x = [] # tablica x_i
d1 = [] # tablica d_i

for i in range(m+1):
    x.append(-a + i * h)
    
for i in range(m+1):
    val = 1 / (h**2) + x[i]**2 / 2 + lam * x[i]**4
    d1.append(val)

mat1 = H1(m) # macierz trójdiagonalna

eig_val_1, eig_vec_1 = np.linalg.eig(mat1)

indices1 = np.argsort(eig_val_1)[:4]
smallest_vals_1 = []
smallest_vecs_1 = []

for idx in indices1:
    smallest_vals_1.append(eig_val_1[idx])
    smallest_vecs_1.append(eig_vec_1[:,idx])

for i in range(len(smallest_vecs_1)):
    plt.plot(x[:1000],smallest_vecs_1[i], label=f"E={smallest_vals_1[i]:.3f}")
plt.legend()
plt.title("Cztery rozwiązania oscylatora anharmonicznego")
plt.show()


# anharmoniczny vs harmoniczny
def H2(n):
    matrix = np.zeros((n,n))
    for i in range(n):
        matrix[i][i] = d2[i+1]
        if i + 1 < n:
            matrix[i + 1][i] = g
            matrix[i][i + 1] = g
    return matrix

d2 = [] # nowa tablica d_i
    
for i in range(m+1):
    # print(i)
    val = 1 / (h**2) + x[i]**2 / 2
    d2.append(val)

mat2 = H2(m)

eig_val_2, eig_vec_2 = np.linalg.eig(mat2)

indices2 = np.argsort(eig_val_2)[:4]
smallest_vals_2 = []
smallest_vecs_2 = []


for idx in indices2:
    smallest_vals_2.append(eig_val_2[idx])
    smallest_vecs_2.append(eig_vec_2[:,idx])

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

k = 0
for i in range(2):
    for j in range(2):
        ax = axs[i, j]
        ax.plot(x[:1000], smallest_vecs_1[k], label=f"lambda = 0.2")
        ax.plot(x[:1000], smallest_vecs_2[k], label=f"lambda = 0")
        ax.set_title(f"E={smallest_vals_1[k]:.3f}")
        ax.legend()
        k+=1

plt.suptitle("Anharmoniczny (lambda = 0.2) vs harmoniczny (lambda = 0)")
plt.tight_layout()
plt.show()
