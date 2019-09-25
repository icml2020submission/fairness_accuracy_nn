import numpy as np
from random import seed, shuffle
import math
import pandas

from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
import seaborn as sns

# TODO: S, add some more interesting examples! add some high-dimensional x and multiclass y

# Our collection of synthetic datasets are housed in this function

def generateSimData(n,sim,visualize_data=False):

    """Generate simulated data

    Args:
        n (int): total sample size
        sim (str): string name  'SJL',
                                'ZafarWWW2017CaseII',
                                'ZafarAISTATS2017'

    Returns:
        x: predictive covariates
        z: (potentially) multivariate categorical protected variable, DUMMY CODED WITH PANDAS, DROP_FIRST = TRUE
        y: univariate categorical response

    """
    # bivariate x = (continuous x1, discrete x2)
    # univariate binary z
    # univariate binary y
    if sim == 'SJL':

        # protected variable z
        p = 1/2
        z = np.random.binomial(1, p, n)
        # continuous x1 and discrete x2
        x1 = np.random.normal(0, 1, n) + z - 4 # standard deviation controls how correlated x1 and z are
        mu = np.exp(-1+np.multiply(x1, z)/2+x1/10+z/6)
        x2 = np.random.poisson(mu)
        x = np.column_stack((x1,x2))
        lincomp = x1+2*x2+z+2.5
        probabilities = np.exp(lincomp)/(1+np.exp(lincomp))
        # binary target
        y = np.random.binomial(1, probabilities)

        # estimate bayes error
        bayeserror = np.mean(y[probabilities <= 0.5] == 1) * np.mean(probabilities <= 0.5) \
                     + np.mean(y[probabilities > 0.5] == 0) * np.mean(probabilities > 0.5)

        # visualize simulation data
        if visualize_data:
            visualizeSJL(x, z, probabilities, lincomp, bayeserror)

    # bivariate x = (continuous x1, discrete x2)
    # bivariate categorical z = (3-level z1, 5-level z2) dummy coded to 2+4 = 6-variate z
    # univariate binary y
    if sim == 'SLmultiZ':

        p = 1/2
        temp = np.random.binomial(1, p, n)
        # continuous x1 and discrete x2
        x1 = np.random.normal(0, 1, n) + temp - 4 # standard deviation controls how correlated x1 and z are
        mu = np.exp(-1+np.multiply(x1, temp)/2+x1/10+temp/6)
        x2 = np.random.poisson(mu)
        x = np.column_stack((x1,x2))
        lincomp = x1+2*x2+temp+2.5
        probabilities = np.exp(lincomp)/(1+np.exp(lincomp))

        # multivariate categorical protected variable z
        z1 = (probabilities <= 0.2) + 2 * np.logical_and(probabilities > 0.2, probabilities <= 0.5) + 3 * (probabilities > 0.5)
        z2 = np.random.choice(5, n)
        z = np.column_stack((pandas.get_dummies(z1, drop_first=True),pandas.get_dummies(z2, drop_first=True)))

        # binary target
        y = np.random.binomial(1, probabilities)


    # bivariate x = (continuous x1, continuous x2)
    # univariate binary z
    # univariate binary y
    elif sim == 'ZafarWWW2017CaseII':

        # Generate data such that a classifier optimizing for accuracy
        # will have disparate false positive rates as well as disparate
        # false negative rates for both groups.

        n_samples = n//4

        cc = [[10, 1], [1, 4]]
        mu1, sigma1 = [2, 3], cc  # z=1, +
        cc = [[5, 2], [2, 5]]
        mu2, sigma2 = [1, 2], cc  # z=0, +

        cc = [[5, 1], [1, 5]]
        mu3, sigma3 = [-5, 0], cc  # z=1, -
        cc = [[7, 1], [1, 7]]
        mu4, sigma4 = [0, -1], cc  # z=0, -

        nv1, X1, y1, z1 = gen_gaussian_diff_size(mu1, sigma1, 1, +1, int(n_samples * 1))  # z=1, +
        nv2, X2, y2, z2 = gen_gaussian_diff_size(mu2, sigma2, 0, +1, int(n_samples * 1))  # z=0, +
        nv3, X3, y3, z3 = gen_gaussian_diff_size(mu3, sigma3, 1, 0, int(n_samples * 1))  # z=1, -
        nv4, X4, y4, z4 = gen_gaussian_diff_size(mu4, sigma4, 0, 0, int(n_samples * 1))  # z=0, -

        # merge the clusters
        x = np.vstack((X1, X2, X3, X4))
        y = np.hstack((y1, y2, y3, y4))
        z = np.hstack((z1, z2, z3, z4))

        # shuffle the data
        perm = list(range(len(x)))
        shuffle(perm)
        x = x[perm]
        y = y[perm]
        z = z[perm]


    # bivariate x = (continuous x1, continuous x2)
    # univariate binary z
    # univariate binary y
    elif sim == 'ZafarAISTATS2017':

        n_samples = n//2  # generate these many data points per class
        disc_factor = math.pi / 4.0  # this variable determines the initial discrimination in the data -- decraese it to generate more discrimination

        def gen_gaussian(mean_in, cov_in, class_label):
            nv = multivariate_normal(mean=mean_in, cov=cov_in)
            X = nv.rvs(n_samples)
            y = np.ones(n_samples, dtype=float) * class_label
            return nv, X, y

        # We will generate one gaussian cluster for each class
        mu1, sigma1 = [2, 2], [[5, 1], [1, 5]]
        mu2, sigma2 = [-2, -2], [[10, 1], [1, 3]]
        nv1, X1, y1 = gen_gaussian(mu1, sigma1, 1)  # positive class
        nv2, X2, y2 = gen_gaussian(mu2, sigma2, 0)  # negative class

        # join the positive and negative class clusters
        X = np.vstack((X1, X2))
        y = np.hstack((y1, y2))

        # shuffle the data
        perm = list(range(0, n_samples * 2))
        shuffle(perm)
        X = X[perm]
        y = y[perm]

        rotation_mult = np.array(
            [[math.cos(disc_factor), -math.sin(disc_factor)], [math.sin(disc_factor), math.cos(disc_factor)]])
        X_aux = np.dot(X, rotation_mult)

        # Generate the protected variable
        x_control = []  # this array holds the sensitive feature value
        for i in range(0, len(X)):
            x = X_aux[i]

            # probability for each cluster that the point belongs to it
            p1 = nv1.pdf(x)
            p2 = nv2.pdf(x)

            # normalize the probabilities from 0 to 1
            s = p1 + p2
            p1 = p1 / s
            p2 = p2 / s

            r = np.random.uniform()  # generate a random number from 0 to 1

            if r < p1:  # the first cluster is the positive class
                x_control.append(1.0)  # 1.0 means its male
            else:
                x_control.append(0.0)  # 0.0 -> female

        z = np.array(x_control)

        x = X

    # bivariate x = (continuous x1, discrete x2)
    # univariate binary z
    # univariate continuous y
    elif sim == 'JLcontinuousY':
        # protected variable z
        p = 1 / 2
        z = np.random.binomial(1, p, n)
        # continuous x1 and discrete x2
        x1 = np.random.normal(0, 1, n) + z + 4
        mu = np.exp(-1 + np.multiply(x1, z) / 2 + x1 / 10 + z / 6)
        x2 = np.random.poisson(mu)
        x = np.column_stack((x1, x2))
        y = 2 * x1 + x2 + z

    # bivariate x = (continuous x1, discrete x2)
    # univariate binary z
    # univariate continuous y
    elif sim == 'GMcontinuousY':
        # protected variable z
        p = 1 / 2
        z = np.random.binomial(1, p, n)
        # continuous x1 and x2
        x1 = np.random.normal(0, 1, n) + z + 4
        mu = -1 + np.multiply(x1, z) / 2 + x1 / 10 + z / 6
        x2 = np.random.normal(0, 1, n) + mu
        x = np.column_stack((x1, x2))
        y = 2 * x1 + x2 + z

        # visualize simulation data
        if visualize_data:
            visualizeContinuousY(x, z, y)

    elif sim == 'gaussian_test':
        cc = [[1, 0], [0, 1]]
        mu1, sigma1 = [-1, 0], cc  # z=1, +
        mu2, sigma2 = [1,0], cc

        n_samples = n // 2

        nv1, X1, y1, z1 = gen_gaussian_diff_size(mu1, sigma1, 1, +1, int(n_samples * 1))  # z=1, +
        nv2, X2, y2, z2 = gen_gaussian_diff_size(mu2, sigma2, 0, +1, int(n_samples * 1))  # z=0, +

        #nv3, X3, y3, z3 = gen_gaussian_diff_size(mu3, sigma3, 1, 0, int(n_samples * 1))  # z=1, -
        #nv4, X4, y4, z4 = gen_gaussian_diff_size(mu4, sigma4, 0, 0, int(n_samples * 1))  # z=0, -

        # merge the clusters
        #x = np.vstack((X1, X2, X3, X4))
        #y = np.hstack((y1, y2, y3, y4))
        #z = np.hstack((z1, z2, z3, z4))

        x = np.vstack((X1, X2))
        y = np.hstack((y1, y2))
        z = np.hstack((z1, z2))

        # shuffle the data
        perm = list(range(len(x)))
        shuffle(perm)
        x = x[perm]
        y = y[perm]
        z = z[perm]

        # change z to pandas data frame, dummy variable
        z = pandas.get_dummies(z, drop_first=True)

    return x, z, y


def gen_gaussian_diff_size(mean_in, cov_in, z_val, class_label, n):

    """
    mean_in: mean of the gaussian cluster
    cov_in: covariance matrix
    z_val: sensitive feature value
    class_label: +1 or 0
    n: number of points
    """

    nv = multivariate_normal(mean=mean_in, cov=cov_in)
    X = nv.rvs(n)
    y = np.ones(n, dtype=float) * class_label
    z = np.ones(n, dtype=float) * z_val  # all the points in this cluster get this value of the sensitive attribute

    return nv, X, y, z


def visualizeContinuousY(x, z, y):

    fig = plt.figure(figsize=(20, 20))  # gives png a*100 by b*100

    x1 = x[:, 0]
    x2 = x[:, 1]

    # visualize distribution of x1 in each level of protected variable z
    fig.add_subplot(131)
    sns.distplot(x1[z == 0], hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3},
                 label='protected z=1')
    sns.distplot(x1[z == 1], hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3},
                 label='protected z=1')
    plt.legend(loc='upper right')
    plt.title('distribution of x1')

    # visualize distribution of x2 in each level of protected variable z
    fig.add_subplot(132)
    sns.distplot(x2[z == 0], hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3},
                 label='protected z=0')
    sns.distplot(x2[z == 1], hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3},
                 label='protected z=1')
    plt.legend(loc='upper right')
    plt.title('distribution of x2')

    fig.add_subplot(133)
    sns.distplot(y[z == 0], hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3},
                 label='protected z=0')
    sns.distplot(y[z == 1], hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3},
                 label='protected z=1')
    plt.legend(loc='upper right')
    plt.title('distribution of y')

    plt.show()

def visualizeSJL(x, z, probabilities, lincomp, bayeserror):

    fig = plt.figure(figsize=(20, 20)) #gives png a*100 by b*100

    x1 = x[:,0]
    x2 = x[:,1]

    # visualize distribution of x1 in each level of protected variable z
    fig.add_subplot(321)
    sns.distplot(x1[z == 0], hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3},
                 label='protected z=1')
    sns.distplot(x1[z == 1], hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3},
                 label='protected z=1')
    plt.legend(loc='upper right')
    plt.title('distribution of x1')


    # visualize distribution of x2 in each level of protected variable z
    fig.add_subplot(322)
    sns.distplot(x2[z == 0], hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3},
                 label='protected z=0')
    sns.distplot(x2[z == 1], hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3},
                 label='protected z=1')
    plt.legend(loc='upper right')
    plt.title('distribution of x2')

    # visualize distribution of probabilities in each level of protected variable z
    fig.add_subplot(323)
    sns.distplot(probabilities[z == 0], hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3},
                 label='protected z=0')
    sns.distplot(probabilities[z == 1], hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3},
                 label='protected z=1')
    plt.legend(loc='upper right')
    plt.title('distribution of true probabilities, bayes error rate =  %.2g' % bayeserror)

    # x1 versus probabilities
    fig.add_subplot(324)
    plt.scatter(x1, probabilities)
    plt.title('x1 versus probabilities')

    # x2 versus probabilities
    fig.add_subplot(325)
    plt.scatter(x2, probabilities)
    plt.title('x2 versus probabilities')


    # visualize distribution of linear component
    fig.add_subplot(326)
    sns.distplot(lincomp, hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3},
                 label='protected z=0')
    plt.title('distribution of linear component')
    plt.show()

