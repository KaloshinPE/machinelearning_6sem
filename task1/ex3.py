import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

Data = [[],[]]
for i in range(500):
    Data[0].append(np.random.uniform(0, 5))
    Data[1].append(Data[0][-1]*0.5 + 1 + np.random.normal(scale=0.2))


a = minimize(lambda a: np.mean(map(lambda i: (Data[1][i] - a[0]*Data[0][i] - a[1])**2, range(len(Data)))), [0, 0])#.x
print a

x = np.linspace(np.array(Data[0]).min(), np.array(Data[0]).max(), 100)
y = [a[0]*xx + a[1] for xx in x]



for i in range(75):
    Data[0].append(np.random.uniform(x[0], x[-1]))
    Data[1].append(-1 + np.random.normal(scale=0.2))

a_mse = minimize(lambda a: np.mean(map(lambda i: (Data[1][i] - a[0]*Data[0][i] - a[1])**2, range(len(Data)))), [0, 0]).x
a_mae = minimize(lambda a: np.mean(map(lambda i: np.abs(Data[1][i] - a[0]*Data[0][i] - a[1]), range(len(Data)))), [0, 0]).x
print a_mse, a_mae

y_mse = [a_mse[0]*xx + a_mse[1] for xx in x]
y_mae = [a_mae[0]*xx + a_mae[1] for xx in x]


plt.scatter(Data[0], Data[1], linewidths=0.5, alpha=0.2)
plt.plot(x, y, label='MSE (not disturbed)', color='r')
plt.plot(x, y_mse, label='MSE', color='m')
plt.plot(x, y_mae, label='MAE', color='y')
plt.legend()
plt.show()