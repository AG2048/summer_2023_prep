import torch
from matplotlib import pyplot as plt

# Key: Z(t+1) = Z(t) + h * AZ(t)

"""
Huxley-Hodgkin equations

https://en.wikipedia.org/wiki/Hodgkin–Huxley_model

dV/dt = (I - g_Na*m^3*h*(V - V_Na) - g_K*n^4*(V - V_K) - g_l*(V - V_l)) / C_m
dm/dt = alpha_m*(1 - m) - beta_m*m
dn/dt = alpha_n*(1 - n) - beta_n*n
dh/dt = alpha_h*(1 - h) - beta_h*h

alpha_m = 0.1 * (25 - V) / (exp((25 - V) / 10) - 1)
beta_m = 4 * exp(-V / 18)
alpha_n = 0.01 * (10-V) / (exp((10 - V) / 10) - 1)
beta_n = 0.125 * exp(-V / 80)
alpha_h = 0.07 * exp(-V / 20)
beta_h = 1 / (exp( (30 - V) / 10) + 1)
"""

# Constants
C_m = 1
g_Na = 120
V_Na = 115
g_K = 36
V_K = -12
g_l = 0.3
V_l = 10.6
I = 50

dt = 0.01

# 8 by 1 vector, V m n h
Z = torch.zeros(4, dtype=torch.float32).view(4, 1)

# Generate function f(Z) = V' m' n' h'
# V' = (I - g_Na*m^3*h*(V - V_Na) - g_K*n^4*(V - V_K) - g_l*(V - V_l)) / C_m
# m' = alpha_m*(1 - m) - beta_m*m
# n' = alpha_n*(1 - n) - beta_n*n
# h' = alpha_h*(1 - h) - beta_h*h

def f(Z):
    V_prime = (I - g_Na * Z[1] ** 3 * Z[3] * (Z[0] - V_Na) - g_K * Z[2] ** 4 * (Z[0] - V_K) - g_l * (Z[0] - V_l)) / C_m
    m_prime = (0.1 * (25 - Z[0]) / (torch.exp((25 - Z[0]) / 10) - 1) * (1 - Z[1]) - 4 * torch.exp(-Z[0] / 18) * Z[1])
    n_prime = 0.01 * (10 - Z[0]) / (torch.exp((10 - Z[0]) / 10) - 1) * (1 - Z[2]) - 0.125 * torch.exp(-Z[0] / 80) * Z[2]
    h_prime = (0.07 * torch.exp(-Z[0] / 20) * (1 - Z[3]) - 1 / (torch.exp((30 - Z[0]) / 10) + 1) * Z[3])
    return torch.cat((V_prime, m_prime, n_prime, h_prime), 0).view(4, 1)
V_list = []
time_list = []
t = 0
m_list = []
n_list = []
h_list = []
while t < 100:
    Z += dt * f(Z)
    V_list.append(Z[0].item())
    m_list.append(Z[1].item())
    n_list.append(Z[2].item())
    h_list.append(Z[3].item())
    time_list.append(t)
    t += dt

plt.subplot(2, 1, 1)
plt.plot(time_list, V_list)
plt.xlabel('time')
plt.ylabel('V')

plt.subplot(2, 1, 2)
plt.plot(V_list, m_list)
plt.xlabel('V')
plt.ylabel('m, n, h')
plt.plot(V_list, n_list)
plt.plot(V_list, h_list)
plt.legend(['m', 'n', 'h'])


plt.show()