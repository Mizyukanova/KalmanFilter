import math
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

def ReadMatrixFromFile(filename):
    matrix = np.loadtxt(filename, delimiter=' ')
    return matrix

# Переходная матрица состояний Ф(k+1,k) mxm
Fi = ReadMatrixFromFile('matrix_Fi.txt')
# Переходная матрица возмущений Г(k+1,k) mxp
Gamma = ReadMatrixFromFile('matrix_Gamma.txt')
# Переходная матрица управления Psi(k+1,k) mxr
Psi = ReadMatrixFromFile('matrix_Psi.txt')
# Матрица измерений Н(k+1) nxm
Eta = ReadMatrixFromFile('matrix_H.txt')
# Начальные условия системы
m_x0 = ReadMatrixFromFile('vector_m_x0.txt')
m_x0 = np.asmatrix(m_x0).transpose()
V_x0 = ReadMatrixFromFile('matrix_V_x0.txt')

N = 101
k = np.arange(0,N)

# Математичсекое ожидание и дисперсии
mu = 0
sigma_w = 0.5
sigma_eta = 0.2
# Вектор математических ожиданий возмущения mx1 
m_w = np.full((Gamma.shape[0],1), mu)
# Ковариационная матрица вектора возмущений
V_w = pow(sigma_w,2)*np.eye(Gamma.shape[0])
# Ковариационная матрица шума измерений
V_eta = sigma_eta*np.array([1, 0.1, 1])*np.eye(Eta.shape[0])
V_eta = np.linalg.matrix_power(V_eta, 2)

#   Обнуление векторов
# Вектор состояния x(k+1) mx1 - N штук
x = np.zeros((Fi.shape[0],N))
# Оценка вектора состояния x^(k+1) mx1 - N штук
x_estimate = np.zeros((Fi.shape[0],N))
# Оценка вектора предсказаний x^(k+1|k) mx1 - N штук
x_estimate_pred = np.zeros((Fi.shape[0],N))
# Вектор измерений z(k+1) nx1 
z = np.zeros((Gamma.shape[0],1))
# Ковариационная матрица ошибки оценивания mxm
V_eps = np.zeros(Fi.shape)
# Вектор следов корреляционной матрицы ошибки оценивания Nx1 
V_eps_traces = np.zeros((1,N))
# Вектор ошибки оценивания Delta(k) Nx1 
Delta = np.zeros((1,N))
# Вектор относительного уровня шума delta(k) 1xN 
delta = np.zeros((1,N))

# Задание начальных условий перед рекурентным алгоритмом
x[:,[0]] = m_x0
x_estimate[:,[0]] = m_x0
V_eps = V_x0
V_eps_traces[:,[0]] = np.trace(V_eps)

for i in range(N-1):
    # Вычисление матрицы фильтр Калмана
    V_eps_pred = Fi*V_eps*Fi.transpose() + Gamma*V_w*Gamma.transpose()
    K = V_eps_pred*Eta.transpose()*np.linalg.inv(Eta*V_eps_pred*Eta.transpose() + V_eta)
    V_eps = (np.eye(K.shape[0]) - K*Eta)*V_eps_pred

    # След корреляционной матрицы ошибки оценивания 
    V_eps_traces[:,[i+1]] = np.trace(V_eps)

    # Генерация случайных независимых чисел, имеющих нормальный закон распределения mx1
    w = np.random.normal(mu, sigma_w, (Gamma.shape[0],1))
    # Генерация случайных независимых чисел, имеющих нормальный закон распределения nx1
    eta = np.array([ np.random.normal(mu, sigma_eta), 0.1*np.random.normal(mu, sigma_eta), np.random.normal(mu, sigma_eta) ])
    eta = np.asmatrix(eta).transpose()
    # Генерация вектора управления rx1
    u = np.array([2*math.sin(0.3*i), 0, 0.2*i])
    u = np.asmatrix(u).transpose()
    # Вектор измерения z(k+1) nx1  
    z = Eta.dot(x[:,[i]]) + eta
    
    # Вычисление предсказания и оценки вектора состояния
    x_estimate_pred[:,[i+1]] = Fi.dot(x_estimate[:,[i]]) + Gamma.dot(m_w) + Psi.dot(u)
    x_estimate[:,[i+1]] = x_estimate_pred[:,[i+1]] + K.dot(z - Eta.dot(x_estimate_pred[:,[i+1]]))
   
    # Вектор состояния x(k+1) mx1
    x[:,[i+1]] = Fi.dot(x[:,[i]]) + Gamma.dot(w) + Psi.dot(u)

    # Вектор ошибки оценивания Delta(k) 1xN 
    Delta[:,[i]] = LA.norm(x_estimate[:,[i]] - x[:,[i]])/LA.norm(x[:,[i]]) 
    
    # Вектор относительного уровня шума delta(k) 1xN 
    delta[:,[i+1]] = LA.norm(eta)/LA.norm(z-eta) 

#   Построение графиков
# Вывод проекций вектора состояния
for j in range(x.shape[0]):
    plt.title('Проекция вектора состояния и его оценки №' + str(j+1))
    plt.plot(k,x[[j],:].transpose(),'-b', label='x(k)')
    plt.plot(k,x_estimate[[j],:].transpose(),':r', label='оценка x(k)')
    plt.legend()
    plt.grid()
    plt.xlabel('k')
    plt.xlim(1, N-1)   
    plt.xticks(np.arange(1, N, 11))
    plt.show()

# Вывод кривой ошибки оценивания
plt.title('Ошибка оценивания - Delta')
plt.plot(k,Delta[[0],:].transpose(),'-b', label='Delta(k)')
plt.legend()
plt.grid()
plt.xlabel('k')
plt.xlim(1, N-1)   
plt.xticks(np.arange(1, N, 11))
plt.show()

# Общие ошибки оценивания при l=1
Delta_general = 1/(N-1)*np.sum(Delta)
print('Общая ошибка оценивания при l=1 равна ', Delta_general)
# Общие ошибки оценивания при l=20
Delta_general = 1/(N-20)*np.sum(Delta)
print('Общая ошибка оценивания при l=20 равна ', Delta_general)

# Вывод кривой ошибки оценивания
plt.title('Ошибка оценивания - delta')
plt.plot(k,delta[[0],:].transpose(),'-b', label='delta(k)')
plt.legend()
plt.grid()
plt.xlabel('k')
plt.xlim(1, N-1)   
plt.xticks(np.arange(1, N, 11))
plt.show()

# Обусловленность матрицы Eta^T*Eta
print('Обусловленность матрицы равна',LA.cond(Eta.transpose().dot(Eta)))

# Вывод следов корреляционной матрицы ошибки оценивания
plt.title('Следы матрицы ошибки оценивания V_eps')
plt.plot(k,V_eps_traces[[0],:].transpose(),'-b', label='V_eps_traces(k)')
plt.legend()
plt.grid()
plt.xlabel('k')
plt.xlim(1, N-1)   
plt.xticks(np.arange(1, N, 11))
plt.show()

# Доп графики
plt.plot(k,Delta[[0],:].transpose(),'-b', label='Delta(k)')
plt.plot(k,delta[[0],:].transpose(),':r', label='delta(k)')
plt.legend()
plt.grid()
plt.xlabel('k')
plt.xlim(1, N-1)   
plt.xticks(np.arange(1, N, 11))
plt.show()