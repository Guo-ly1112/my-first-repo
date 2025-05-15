import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_blobs

def random_data():
    # 创建一个随机的二分类数据集
    X, y = make_blobs(n_samples=100, centers=2, random_state=42)
    return X, y


def iris_data():
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return X, y


# SVM分类器训练函数
def train_svm_classifier(X, y, kernel='linear', C=1.0):
    clf = svm.SVC(kernel=kernel, C=C)
    clf.fit(X, y)
    return clf


# 可视化SVM分类结果
# 可视化SVM分类结果
def plot_svm_results(X, y, clf, title=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

    # 绘制分割面
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    # 绘制支持向量
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none',
               edgecolors='k')

    # 计算margin边界
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = YY - np.sqrt(1 + clf.coef_[0][0] ** 2) * margin
    yy_up = YY + np.sqrt(1 + clf.coef_[0][0] ** 2) * margin

    ax.plot(XX[0], yy_down[0], 'k--', alpha=0.5)
    ax.plot(XX[0], yy_up[0], 'k--', alpha=0.5)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.show()




def main():
    # (1) 使用不同核函数和平衡系数C运行分类器
    kernels = ['linear', 'rbf', 'poly']
    Cs = [0.1, 1, 10]

    for kernel in kernels:
        for C in Cs:
            # 对每个数据集运行SVM分类器
            for data_func in [random_data, iris_data]:  # 假设toy_data()不是必要的，或者已经定义好了
                X, y = data_func()
                clf = train_svm_classifier(X, y, kernel=kernel, C=C)
                print(
                    f"Kernel: {kernel}, C: {C}, Dataset: {data_func.__name__}, Accuracy: {accuracy_score(y, clf.predict(X))}")

                # (2) 对iris_data进行划分并找到最佳SVM超参数
    X_iris, y_iris = iris_data()
    X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.1, random_state=42)
    best_kernel = None
    best_C = None
    best_score = 0

    for kernel in kernels:
        for C in Cs:
            clf = train_svm_classifier(X_train, y_train, kernel=kernel, C=C)
            score = accuracy_score(y_test, clf.predict(X_test))
            if score > best_score:
                best_score = score
                best_kernel = kernel
                best_C = C
                best_clf = clf

    print(f"Best kernel: {best_kernel}, Best C: {best_C}, Best Score: {best_score}")

    # (3) 打印SVM分类器最终获得的支持向量集
    print("Support vectors:")
    print(best_clf.support_vectors_)

    # 可视化random_data上的分类结果
    X_random, y_random = random_data()
    best_random_clf = train_svm_classifier(X_random, y_random, kernel=best_kernel, C=best_C)
    plot_svm_results(X_random, y_random, best_random_clf,
                     title=f"SVM Classification with Best Params (Kernel: {best_kernel}, C: {best_C})")


# 运行主函数
if __name__ == "__main__":
    main()

