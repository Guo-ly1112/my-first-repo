#实现一元线性回归和梯度下降算法

import numpy as np
import matplotlib.pyplot as plt

true_w = 2
true_b = 1
data_lens = 100
np.random.seed(123)

def get_data():
    """生成伪数据集"""
    data_x = np.random.randint(-1,8,size=data_lens)
    data_x = data_x.astype(np.float32)
    data_true_y = [true_w * x +true_b for x in data_x]
    data_y = [y + np.random.randn() for y in data_true_y]

    return data_x,data_y,data_true_y


def draw_data(x,y):
    plt.scatter(x,y)
    plt.xlabel("X")
    plt.ylabel("Y")


def draw_hypeplane(w,b,X,label):
    Y = [w * x + b for x in X]
    plt.plot(X,Y,label=label)


def init_params():
    w = 0.0
    # w = np.random.randn()
    b = 0.0
    return w,b


def predict(w,b,x):
    return w*x+b

def train(X,Y,lr = 0.01):
    # init: w0, b0
    w, b = init_params()    #theta0
    n = len(X)

    R_thetas = []
    R_thetas.append(np.mean([np.square(predict(w, b, X[i])-Y[i]) for i in range(n)]))     #theta0


    # iter：
    for i in range(20):
        w = w - lr*2*np.mean([(predict(w,b,X[j])-Y[j])*X[j] for j in range(n)])
        b = b - lr*2*np.mean([(predict(w,b,X[j])-Y[j]) for j in range(n)])

        R_thetas.append(np.mean([np.square(predict(w,b,X[i])-Y[i]) for i in range(n)]))

    #end
    return w, b, R_thetas


if __name__ == "__main__":
    X, Y = get_data()[:2]
    w, b, R_thetas = train(X, Y)
    print("w: ", w, "b: ", b)
    print("R(theta)s: ", R_thetas)

    draw_data(X, Y)
    draw_hypeplane(true_w, true_b, X, "true")
    draw_hypeplane(w, b, X, "train")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(R_thetas, label="R(theta)s")
    plt.xlabel("epoch")
    plt.ylabel("R(theta)")
    plt.legend()
    plt.show()



























