import numpy as np
import matplotlib.pyplot as plt
x = np.array([1,2,3,4,5])
y = np.array([5,7,9,1,13])

def grad_des(x,y):
    m = c = 2
    lr = 0.01
    n = len(x)

    for i in range(1000):
        y_hat = m*x + c
        err = y - y_hat
        cost = (1/n)*(sum(err**2))

        m_grad = (2/n) * x * sum(err)
        c_grad = (2/n) * sum(err)

        m += lr * m_grad
        c += lr * c_grad
        if not i%100:
            print(f'COST : {cost}')
        plt.plot(x, y_hat)
    plt.scatter(x, y)
    plt.show()
grad_des(x, y)
