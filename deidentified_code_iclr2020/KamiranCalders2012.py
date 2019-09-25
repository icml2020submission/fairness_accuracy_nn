# Kamiran and Calders (2012) describe three data preporcessing techniques to remove discrimation BEFORE a classifier is learned
# 1. Suppression of the sensitive attribute
# 2. Massaging the dataset by changing class labels. Implemented as uniform_sampling below.
# 3. Reweighing or resampling the data without relabeling instances. Implemented as massage below.
# Methods 2 and 3 are limited to one binary protected attribute, and binary target y

import torch
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

def Kamiran_wrapper(x_train,y_train,z_train,xl):
# Both Kamiran methods -- uniform_sampling and flip class labels -- are limited to univariate binary protected z and univariate binary y

    # first change variable type so KC functions are happy
    # must add sensitive attribute to dataFrame X
    temparray = np.column_stack((x_train[:, 0, :], z_train[:, 0]))
    X = pd.DataFrame(data=temparray)
    X.columns = xl+['z']

    X_prime_KCunif, y_prime_KCunif = uniform_sample(X, y_train[:, 0, 0], 'z', 1, 1)
    x_train_KCunif_torch = torch.tensor(X_prime_KCunif[xl].values)
    y_train_KCunif_torch = torch.tensor(y_prime_KCunif)
    # todo: S, I added the following line, so that z also still has the same dimension
    z_train_KCunif_torch = torch.tensor(X_prime_KCunif['z'].values)

    X_prime_KCmassage, y_prime_Kmassage = massage(X, y_train[:, 0, 0], 'z', 1, 1)
    x_train_KCmassage_torch = torch.tensor(X_prime_KCmassage[xl].values)
    y_train_KCmassage_torch = torch.tensor(y_prime_Kmassage)

    return x_train_KCunif_torch, \
           y_train_KCunif_torch.unsqueeze(1), \
           z_train_KCunif_torch.unsqueeze(1), \
           x_train_KCmassage_torch, \
           y_train_KCmassage_torch.unsqueeze(1)


def uniform_sample(X, y, S, b, d):
    """Implementation of the 'uniform sampling' data preprocessing technique
    given by Algorithms 4 from Kamiran and Calders (2012)

    Generate a new training dataset by uniformly sampling from the
    input dataset, with the number of examples drawn from each group/class combo
    chosen to make discKC(X_prime, y) = 0

    Args:
        X (DataFrame): Training data
        y (list): Binary class labels for training data
        S (str): Name of sensitive attribute (binary)
        b: Protected value for sensitive attribute, 0 or 1
        d: Desired class, 0 or 1

    Returns:
        DataFrame: New training data
        list: New binary class labels
    """

    X['label'] = y

    W = pd.DataFrame({'group': [1, 1, 0, 0], 'label': [1, 0, 1, 0]})

    # Calculate weight for each combination of sensitive attribute and class,
    # given by the number of examples in each group divided by the number
    # that should be in each group if the data were non-discriminatory
    # NOTE: Algorithm 4 in the paper actually usees a denominator that appears to be wrong...
    weights = [[len(X[X[S] == s]) * len(X[X['label'] == c]) / float(len(X)*0.25)
                # / float(len(X) * len(X[(X[S] == s) & (X['label'] == c)])) \
                for c in [1, 0]]  for s in [1, 0]]

    sizes = [[len(X[(X[S] == s) & (X['label'] == c)]) for c in [1, 0]] for s in [1, 0]]

    W['weight'] = [i for j in weights for i in j]
    W['size'] = [i for j in sizes for i in j]
    W = W.assign(num = lambda x: x.size * x.weight)

    # Divide the data into the four groups based on class/group
    dp = X[(X[S] == b) & (X['label'] == d)]
    dn = X[(X[S] == b) & (X['label'] != d)]
    fp = X[(X[S] != b) & (X['label'] == d)]
    fn = X[(X[S] != b) & (X['label'] != d)]

    # Uniformly sample from each group
    dp = dp.sample(n = W.loc[(W['group'] == b) & (W['label'] == d), 'num'].iloc[0].astype(int), replace = True)
    dn = dn.sample(n = W.loc[(W['group'] == b) & (W['label'] != d), 'num'].iloc[0].astype(int), replace = True)
    fp = fp.sample(n = W.loc[(W['group'] != b) & (W['label'] == d), 'num'].iloc[0].astype(int), replace = True)
    fn = fn.sample(n = W.loc[(W['group'] != b) & (W['label'] != d), 'num'].iloc[0].astype(int), replace = True)

    X_prime = pd.concat([dp, dn, fp, fn])
    X.drop('label', axis = 1, inplace = True)
    y_prime = X_prime['label'].tolist()
    X_prime = X_prime.drop('label', axis = 1)

    return(X_prime, y_prime)

def massage(X, y, S, b, d):
    """Implementation of the 'massaging' data preprocessing technique
    given by Algorithms 1 & 2 from Kamiran and Calders (2012)

    Flip the class labels of M pairs of examples in order where M
    is chosen explicitly to make the discKC(X, y) = 0. We choose the examples
    for which we flip labels using a ranker, here the ranker is a
    Gaussian Naive Bayes classifier.

    Args:
        X (DataFrame): Training data
        y (list): Binary class labels for training data
        S (str): Name of sensitive attribute (binary)
        b: Protected value for sensitive attribute, 0 or 1
        d: Desired class, 0 or 1

    Returns:
        DataFrame: Training data (identical instances to X, but reordered)
        list: New binary class labels
    """

    # Learn R, a Gaussian NB classifier which will act as a ranker
    R = GaussianNB()
    probas = R.fit(np.asarray(X), y).predict_proba(X)

    # Create a df with training data, labels, and desired class probabilities
    X['class'] = y
    X['prob'] = [record[d] for record in probas]

    # Promotion candidates sorted by descending probability of having desired class
    pr = X[(X[S] == b) & (X['class'] != d)]
    pr = pr.sort_values(by = 'prob', ascending = False)

    # Demotion candidates sorted by ascending probability
    dem = X[(X[S] != b) & (X['class'] == d)]
    dem = dem.sort_values(by = 'prob', ascending = True)

    # Non-candidates
    non = X[((X[S] == b) & (X['class'] == d)) | ((X[S] != b) & (X['class'] != d))]

    # Calculate the discrimination in the dataset
    disc = discKC(X, y, S, b, d)

    # Calculate M, the number of labels which need to be modified
    M = (disc * len(X[X[S] == b]) * len(X[X[S] != b])) / float(len(X))
    M = int(M)

    # Flip the class label of the top M objects of each group
    # i.e. M pairs swap labels, where M is chosen to make discKC = 0
    c = pr.columns.get_loc("class")
    pr.iloc[:M, c] = d
    dem.iloc[:M, c] = 1 - d

    X.drop(['class', 'prob'], axis = 1, inplace = True)
    X_prime = pd.concat([pr, dem, non])
    y_prime = X_prime['class'].tolist()
    X_prime = X_prime.drop(['class', 'prob'], axis = 1)

    return(X_prime, y_prime)

def discKC(X, y, S, b, d):
    """Implementation of the measure of discrimination in a labeled dataset
    or in predictions from a classifier from Definitions 1 and 2 in
    Kamiran and Calders (2012)

    This gives the difference in the probability of being in the desired class
    between the group with the protected value of the sensitive attribute
    (the marginalized group) and the favoured group. The desired class can
    be given by labeled training data, or by the prediction from a classifier.

    Args:
        X (DataFrame): Data
        y (list): Binary class labels or predictions for data
        S (str): Name of sensitive attribute (binary)
        b: Protected value for sensitive attribute, 0 or 1
        d: Desired class, 0 or 1
    """

    D = X.copy()
    D['class'] = y

    # Fraction of examples with non-protected value for S in desired class d
    disc = len(D[(D[S] != b) & (D['class'] == d)]) / float(len(D[D[S] != b])) \
           - len(D[(D[S] == b) & (D['class'] == d)]) / float(len(D[D[S] == b]))

    return(disc)