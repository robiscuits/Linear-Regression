import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from math import sqrt


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_dataset(filename):
    dataset = pd.read_csv(filename)
    del dataset["IDNO"]
    return dataset.values


def print_stats(dataset, col):
    v = dataset[:,col]
    length = len(v)
    avg = round(sum(v)/length, 2)
    sd = round(sqrt(sum((v-avg)**2)/(length-1)),2)
    print("""{}\n{}\n{}""".format(length,avg,sd))
    pass


def regression(dataset, cols, betas):
    vs = dataset[:,cols]
    vs = np.insert(vs,0,np.ones(len(vs)), axis = 1)
    vs = np.matrix(vs)
    y = dataset[:,0]
    betas = np.array(betas)
    n = len(dataset)
    X = vs.dot(betas.T)
    a = (X-y)
    return np.square(a).sum()/n

def gradient_descent(dataset, cols, betas):
    n = len(dataset)
    grads = np.zeros(n)
    vs = dataset[:,cols]
    vs = np.insert(vs,0,np.ones(len(vs)), axis = 1)
    vs = np.matrix(vs)
    y = dataset[:,0]
    betas = np.array(betas)
    X = vs.dot(betas.T)
    grad_mult = (X-y)
    for i in range(len(betas)):
        der = 2*(grad_mult.dot(vs[:,i]))/n
        grads[i] = der
    return grads[:len(betas)]


def iterate_gradient(dataset, cols, betas, T, eta):
    new_betas = betas
    for i in range(T+1):
        curgrad = gradient_descent(dataset, cols, new_betas)
        curmse = regression(dataset, cols, new_betas)
        if i != 0:
            print("{} {}".format(i, "%.2f" % curmse)," ".join(("%.2f" % beta) for beta in new_betas))
        new_betas = new_betas - (eta*curgrad)
    pass


def compute_betas(dataset, cols):
    y = dataset[:,0]
    X = dataset[:,cols]
    X = np.insert(X,0,np.ones(len(dataset)), axis = 1)
    inverse = np.linalg.inv((X.T.dot(X)))
    step2 = inverse.dot(X.T)
    step3 = step2.dot(y)
    betas = list(step3)
    mse = regression(dataset, cols, betas)
    return (mse, *betas)


def predict(dataset, cols, features):
    betas = list(compute_betas(dataset,cols)[1:])
    betas = np.array(betas)
    features.insert(0,1)
    features = np.array(features)
    betas = betas.T
    result = betas.dot(features)
    return result


def synthetic_datasets(betas, alphas, X, sigma):
    linear = np.zeros((len(X), 2))
    quad = np.zeros((len(X), 2))
    b0 = betas[0]
    b1 = betas[1]
    a0 = alphas[0]
    a1 = alphas[1]
    for i in range(len(X)):
        zl = np.random.normal(0,sigma,len(X))
        yi_l = b0 + (X[i][0]*b1) + zl[i]
        linear[i][0] = yi_l
        linear[i][1] = X[i][0]
        
        z_q = np.random.normal(0,sigma,len(X))
        yi_q = a0 + (a1*(X[i][0]**2))+z_q[i]
        quad[i][0] = yi_q
        quad[i][1] = X[i][0]
    return (linear,quad)


def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')
        
    X = np.random.randint(-100,100,(1000,1))
    betas = np.random.randint(1,10,(1,2))[0]
    alphas = np.random.randint(1,10,(1,2))[0]
    powers = list(np.arange(-4,6))
    bases = np.ones(10)
    bases = bases*10
    sigmas = np.power(bases, powers)
    datasets = {}
    sigma_linear = {}
    sigma_quad = {}
    for sigma in sigmas:
        datasets[sigma] = synthetic_datasets(betas, alphas, X, sigma)
        i = 0
        for dataset in datasets[sigma]:
            mse = compute_betas(dataset, cols = [1])[0]
            if i == 0:
                sigma_linear[sigma] = mse
            else:
                sigma_quad[sigma] = mse
            i+=1
    fig, ax = plt.subplots()
    linear = sorted(sigma_linear.items())
    x, y = zip(*linear)
    plt.plot(x,y, marker = "o", label = "linear")
    quad = sorted(sigma_quad.items())
    x, y = zip(*quad)
    plt.plot(x,y, marker = "o", label = "quadratic")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Sigma")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig("mse.pdf")


if __name__ == '__main__':
    ### DO NOT CHANGE THIS SECTION ###
    plot_mse()
