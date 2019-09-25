import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def _add_text(ax,yaxis,xaxis,proj_loss_weights):

    # sz = yaxis.shape
    # assert(len(sz)==2)
    #
    # nr_columns = sz[1]
    # for i in range(nr_columns):
    #     for j,v in enumerate(yaxis[:,i]):
    #         cur_x = xaxis[j]
    #         cur_y = v
    #         cur_pl = proj_loss_weights[j]
    #         ax.text(cur_x, cur_y*1.05, '{:.4f}'.format(cur_pl), horizontalalignment='left', size='small', color='black')

    for j,v in enumerate(yaxis):
        cur_x = xaxis[j]
        cur_y = v
        cur_pl = proj_loss_weights[j]
        ax.text(cur_x, cur_y*1.05, '{:.4f}'.format(cur_pl), horizontalalignment='left', size='small', color='black')


def _has_all_keys(d,keys):

    if type(keys)!=type(['']):
        keys = [keys]

    for k in keys:
        if not (k in d):
            return False
    return True

# plots statistical measures of fairness versus criterion loss (c_loss) simultaneously for in-sample and out-of-sample
# TODO: set keys_yaxis to be one key at a time, otherwise issues with yaxis scale
def plot_sweep_results(proj_loss_weights,sweep_results_train, sweep_results_test, key_xaxis, keys_yaxis,save_filename=None):
    """

    :param proj_loss_weights:
    :param sweep_results_train:
    :param sweep_results_test:
    :param key_xaxis: a string that is compatible with debiasing.populate_sweep_results, e.g. 'c_loss'
    :param keys_yaxis: a string that is compatible with debiasing.populate_sweep_results, e.g. 'proj_loss'
    :param save_filename:
    :return:
    """

    has_all_x_keys_train = _has_all_keys(sweep_results_train,key_xaxis)
    has_all_y_keys_train = _has_all_keys(sweep_results_train,keys_yaxis)
    has_all_x_keys_test = _has_all_keys(sweep_results_test,key_xaxis)
    has_all_y_keys_test = _has_all_keys(sweep_results_test,keys_yaxis)


    if not (has_all_x_keys_train and has_all_y_keys_train and has_all_x_keys_test and has_all_y_keys_test):
        print('INFO: Not all keys available. Ignoring plot.')
        print('INFO: Desired keys were x={}; y={}'.format(key_xaxis,keys_yaxis))
        print('INFO: available train keys are {}'.format(sweep_results_train.keys()))
        print('INFO: available test keys are {}'.format(sweep_results_test.keys()))
        return

    n_train= len(sweep_results_train[key_xaxis])
    n_test= len(sweep_results_test[key_xaxis])

    xaxis_train = np.array(sweep_results_train[key_xaxis])
    xaxis_test = np.array(sweep_results_test[key_xaxis])

    for k in keys_yaxis:

        # create Tidy ("long-form") dataframe where each column is a variable and each row is an observation
        data = pd.DataFrame({key_xaxis: np.array(sweep_results_train[key_xaxis] + sweep_results_test[key_xaxis]),
                             k: np.array(sweep_results_train[k] + sweep_results_test[k]),
                            'sample': ['Train'] * n_train +['Test'] * n_test},
                            index = np.concatenate((proj_loss_weights,proj_loss_weights)))
        plt.clf()
        ax = sns.lineplot(x=key_xaxis,y=k,hue='sample',palette="tab10", style='sample',linewidth=2.5, markers=True, data=data)

        yaxis_train = np.array(sweep_results_train[k])
        yaxis_test = np.array(sweep_results_test[k])

        # now add the text
        _add_text(ax=ax,xaxis=xaxis_train,yaxis=yaxis_train,proj_loss_weights=proj_loss_weights)
        _add_text(ax=ax,xaxis=xaxis_test,yaxis=yaxis_test,proj_loss_weights=proj_loss_weights)

        # plt.ylim(np.min([np.percentile(yaxis_trian,5),np.percentile(yaxis_test,5)]),
        #          np.max([np.percentile(yaxis_trian, 95), np.percentile(yaxis_test, 95)]))

        if not (save_filename is None):
            plt.savefig('{}_{}.pdf'.format(save_filename,k))

    return ax

def create_animated_gif(filter,out_filename):
    cmd = 'convert -delay 75 {} {}'.format(filter,out_filename)

    if os.path.isfile(out_filename):
        os.remove(out_filename)

    print('Creating animated gif: {}'.format(out_filename))
    os.system(cmd)

def visualize_labels(x,y,z=None,title=None,xrange_lim=None,yrange_lim=None):

    #plt.clf()

    sz = x.shape
    if len(sz)<2:
        raise ValueError('Assuming two-dimensional feature data; as columns')
    if sz[1]!=2:
        raise ValueError('Assuming two-dimensional feature data; as columns')

    # convert to pandas dataframe
    px = pd.DataFrame(data=x,columns=['x1','x2'])

    if z is None:
        ax = sns.scatterplot(x='x1', y='x2', data=px, hue=y.reshape(-1))
    else:
        ax = sns.scatterplot(x='x1', y='x2', data=px, hue=y.reshape(-1), style=z.reshape(-1))

    if title is not None:
        plt.title(title)

    if xrange_lim is not None:
        plt.xlim(xrange_lim[0],xrange_lim[1])
    if yrange_lim is not None:
        plt.ylim(yrange_lim[0], yrange_lim[1])

    xrange = ax.get_xlim()
    yrange = ax.get_ylim()

    #plt.show()
    plt.close()

    return xrange,yrange
