import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF
import torch

def categories(series):
    return range(int(series.min()), int(series.max()) + 1)


def r_by_c(df, col1, col2):
    df_col1, df_col2 = df[col1], df[col2]

    # pad the categories in case there is no shows in one category entirely
    if len(categories(df_col1)) == 1:
        cat1s = list(categories(df_col1))
        cat1s.append(max(df_col1)+1)
    else:
        cat1s = categories(df_col1)

    if len(categories(df_col2)) == 1:
        cat2s = list(categories(df_col2))
        cat2s.append(max(df_col2)+1)
    else:
        cat2s = categories(df_col2)

    result = [[sum((df_col1 == cat1) & (df_col2 == cat2))
               for cat2 in cat2s]
              for cat1 in cat1s]
    # pad zero entries of the contingency table with 0.25
    result = [[x + 0.25 * (x == 0) for x in result[i]] for i in range(len(result))]
    return result


def stratified_tables(df,col1,col2,strat_col):

    tables = []
    for cat in categories(df[strat_col]):
        table_cat = r_by_c(df[df[strat_col]==cat],col1,col2)
        tables.append(table_cat)

    return tables


def _fix_data_dim_format(y):
    """
    Standard format in pytorch is BxCxdim; this converts the format by removing channel always one for us and squeezing dimensions
    :param y:
    :return:
    """
    sz = y.shape
    if len(sz)>1:
        if sz[1]!=1:
            raise ValueError('Channel dimension needs to be one so that no information is lost')

        #return (y[:,0,...].squeeze()).astype('int64')
        return y[:,0,...]
    else:
        return y


# this is limited to a DISCRETE conditioning set
def assessfairness(yhat, y, z):

    # TODO: extend all variables to be ordinal, there are more powerful tests, note the generalization to I x J x K tables https://onlinecourses.science.psu.edu/stat504/node/113/
    """
    :param yhat: nominal categorical, coded as array of integers
    :param y: if binary, let 1 denote "favorable" outcome. this is for equality of opportunity calculation
    :param z:  UNIVARIATE nominal categorical
    :return: dictionary of fairness measures
    """

    # initialize dictionary to hold fairness measures
    fairness_measures = dict()

    z_dim = z.shape[2]
    if z_dim!=1:
        print('INFO: only univariate z is currently supported. Ignorning the fairness measures')
        return fairness_measures

    # create data frame
    df = pd.DataFrame({'yhat': _fix_data_dim_format(yhat), 'z': _fix_data_dim_format(z), 'y': _fix_data_dim_format(y)})

    # assessing demographic parity via chi-squared test for independence
    # null hypothesis states yhat and z are (unconditionally) independent
    # we want the chi-squared test statistic to be SMALL
    # table = r_by_c(df,'yhat','z')
    # chi2, p, dof, ex = scs.chi2_contingency(table)
    # fairness_measures['demographic_parity_pvalue'] = p

    # Equalized odds of Hardt et. al 2016
    # Conditional parity between yhat and z conditional on y
    # This will be accomplished via Breslow-Day, possibly followed by CMH test
    # null hypothesis states yhat and z are independent conditional on y
    tables = stratified_tables(df, 'yhat', 'z', 'y')

    # pass in stratified tables
    st = sm.stats.StratifiedTable(tables)

    # calculate breslow-day pvalue to first test homogeneity of odds ratios
    breslow_day_statistic, breslow_day_pvalue = st.test_equal_odds().statistic, st.test_equal_odds().pvalue

    if breslow_day_pvalue > 0.05:  # if null of equal ORs is "accepted", then proceed to CMH test
        cmh_statistic, cmh_pvalue = st.test_null_odds().statistic, st.test_null_odds().pvalue
        fairness_measures['equalized_odds_CMHp'] = cmh_pvalue
        fairness_measures['equalized_odds_CMHstat'] = cmh_statistic
        fairness_measures['equalized_odds_BDp'] = np.nan
        fairness_measures['equalized_odds_BDstat'] = np.nan
    else:
        fairness_measures['equalized_odds_BDp'] = breslow_day_pvalue
        fairness_measures['equalized_odds_BDstat'] = breslow_day_statistic
        fairness_measures['equalized_odds_CMHp'] = np.nan
        fairness_measures['equalized_odds_CMHstat'] = np.nan

    return fairness_measures


def mean_variance_uncond_ind(phat,z):
    """

    :param phat:
    :param z: univariate categorical coded as integers
    :return: mean variance statistic (equation 2.3) in Cui and Zhong 2017, "A Distribution-Free Test of Independence and Its Application to Variable Selection"
    """
    uncond_ecdf = ECDF(phat)(phat)
    setz = np.unique(z, axis=0)

    temp = [((z == setz[i]).mean() * ((ECDF(phat[z == setz[i]])(phat) - uncond_ecdf) ** 2)).sum() for i in range(setz.shape[0])]

    return sum(temp)

def fairness_measures(phat,y,z,propensity_scores):

    # TODO: extend to case of ordinal categorical variables. Not sure there is an existing extension
    """
    phat, y, and z are for the same sample. all categorical variables are treated as nominal rather than ordinal.
    :param phat: {tuple}(n,) fitted probabilities
    :param y: {tuple}(n,) univariate categorical
    :param z {tuple}(n, 1, dimz) : multivariate categorical coded as dummy variables, drop_first = true.
    :return: a measure for phat independent z given y calculated for three options of y:
                -- y = Null  for demographic parity
                -- y = 1 for equal opportunity
                -- y for equalized odds
            in each case, the measure is 0 if and only if independence is achieved.
    """
    fairness_measures = dict()

    # sample estimator of WATE using overlap weights, equation (6) from Li, Lock Morgan, et. al
    w1 = propensity_scores[:, :, 0]  # P(Z=0|X)
    w0 = propensity_scores[:, :, 1]  # P(Z=1|X)
    w1z = w1 * z[:, 0, :]
    w1z_sum = torch.sum(w1z)
    w0znot = w0 * (1 - z[:, 0, :])
    w0znot_sum = torch.sum(w0znot)

    weights = w1z / w1z_sum - w0znot / w0znot_sum
    weighted_h = weights * phat.squeeze(dim=1).clone()
    fairness_measures['causal'] = torch.norm(torch.sum(weighted_h, 0))

    # convert to ndarray
    phat = phat.cpu().numpy()
    phat = phat.flatten()
    y = y.cpu().numpy()
    y = y.flatten()
    z = z.cpu().numpy()

    # turn z into univariate
    # TODO: this is pretty slow! not sure how to make it faster.
    setz = np.unique(z, axis=0)
    if z.ndim != 1:
        z = sum([i*np.all(z == setz[i], axis=2) for i in range(setz.shape[0])]) # this will make z one dimensional
        z = z.flatten()

    # mean-variance statistics
    mvstat = [mean_variance_uncond_ind(phat[y == condition_y], z[y == condition_y]) for condition_y in np.unique(y, axis=0)]
    fairness_measures['mv_EO'] = max(mvstat)
    fairness_measures['mv_EOpp'] = mean_variance_uncond_ind(phat[y == 1], z[y == 1])
    fairness_measures['mv_DP'] = mean_variance_uncond_ind(phat, z)

    return fairness_measures

 # sample estimator of WATE using overlap weights, equation (6) from Li, Lock Morgan, et. al

