from os import path
import pickle
import torch
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

def set_size(width, fraction=1):
    """ Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

def simple_cull(inputPoints, dominates):
    paretoPoints = set()
    candidateRowNr = 0
    dominatedPoints = set()
    while True:
        candidateRow = inputPoints[candidateRowNr]
        inputPoints.remove(candidateRow)
        rowNr = 0
        nonDominated = True
        while len(inputPoints) != 0 and rowNr < len(inputPoints):
            row = inputPoints[rowNr]
            if dominates(candidateRow, row):
                # If it is worse on all features remove the row from the array
                inputPoints.remove(row)
                dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow):
                nonDominated = False
                dominatedPoints.add(tuple(candidateRow))
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            paretoPoints.add(tuple(candidateRow))

        if len(inputPoints) == 0:
            break

    return list(paretoPoints), list(dominatedPoints)

def dominates(row, candidateRow):
    return sum([row[x] <= candidateRow[x] for x in range(len(row))]) == len(row)


for data_num, folder in itertools.product(range(0,3), {"e5c0f79"}):

    if data_num == 0:
        plot_title = 'UCI (gender)'
    elif data_num == 1:
        plot_title = 'UCI (race)'
    elif data_num == 2:  # UCI adult dataset has both gender and race protected attributes
        plot_title = 'Recidivism (race)'

    inputPoints = []
    inputPoints_adv = []

    for mc in range(0, 99):

        name = 'compareMethods_data{}_mc{}.p'.format(data_num, mc)
        filename = path.join(folder, name)

        if path.exists(filename):

            simresults = pickle.load(open(filename, 'rb'))

            closs = torch.stack(simresults['recorded_sweep_results_test']['c_loss'])
            causal = torch.stack(simresults['recorded_sweep_results_test']['causal'])
            closs_causal = torch.stack([closs, causal], dim=1)
            inputPoints.append(closs_causal.tolist())

            if folder == "e5c0f79":
                closs_adv = torch.stack(simresults['adv_recorded_sweep_results_test']['c_loss'])
                causal_adv = torch.stack(simresults['adv_recorded_sweep_results_test']['causal'])
                closs_causal_adv = torch.stack([closs_adv, causal_adv], dim=1)
                inputPoints_adv.append(closs_causal_adv.tolist())

    candidatePareto = list(itertools.chain(*inputPoints))
    candidatePareto_orig = candidatePareto.copy()
    paretoPoints, dominatedPoints = simple_cull(candidatePareto, dominates)
    config_base = pd.DataFrame.from_records(candidatePareto_orig,columns=['BCE','ATO'])
    config_pareto = pd.DataFrame.from_records(paretoPoints,columns=['BCE','ATO']).sort_values('BCE')

    print(plot_title)
    print('# of pareto points {}'.format(paretoPoints.__len__()))
    print('# of dominated points {}'.format(dominatedPoints.__len__()))

    if folder == "e5c0f79":
        candidatePareto_adv = list(itertools.chain(*inputPoints_adv))
        candidatePareto_orig_adv = candidatePareto_adv.copy()
        paretoPoints_adv, dominatedPoints_adv = simple_cull(candidatePareto_adv, dominates)
        config_base_adv = pd.DataFrame.from_records(candidatePareto_orig_adv,columns=['BCE','ATO'])
        config_pareto_adv = pd.DataFrame.from_records(paretoPoints_adv,columns=['BCE','ATO']).sort_values('BCE')

        print(plot_title)
        print('# of adv pareto points {}'.format(paretoPoints_adv.__len__()))
        print('# of adv dominated points {}'.format(dominatedPoints_adv.__len__()))




    # % your document class here
    # \documentclass[10pt]{report}
    # \begin{document}
    #
    # % gives the width of the current document in pts
    # \showthe\textwidth
    #
    # \end{document}

    ## Plotting
    width = 397
    plt.style.use('seaborn')
    nice_fonts = {
            # Use LaTeX to write all text
            "text.usetex": True,
            "font.family": "serif",
            # Use 10pt font in plots, to match 10pt font in document
            "axes.labelsize": 8,
            "font.size": 10,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
    }
    mpl.rcParams.update(nice_fonts)

    if folder == 'e5c0f79':

        fig, (ax1,ax2) = plt.subplots(2,1,sharex='all',sharey='all',figsize=set_size(width,fraction=1))
        sns.regplot(x='BCE', y='ATO', data=config_base, fit_reg=False, label=r'Chebyshev $\hat \theta_n^\lambda$',ax=ax1,color='c',scatter_kws={'s':6})
        ax1.plot(config_pareto['BCE'], config_pareto['ATO'], 'm--', linewidth = 4, label='Pareto frontier', alpha=0.5)
        ax1.set_xlabel(r'$R(\hat \theta_n^\lambda,\hat{P}_{\rm{test}})$')
        ax1.set_ylabel(r'$U(\hat \theta_n^\lambda,\hat{P}_{\rm{test}})$')
        ax1.set_title(plot_title)
        ax1.legend(loc='best')

        sns.regplot(x='BCE', y='ATO', data=config_base_adv, fit_reg=False, label=r'adversarial $\tilde \theta_n^\lambda$',ax=ax2,color='c',scatter_kws={'s':6})
        ax2.plot(config_pareto_adv['BCE'], config_pareto_adv['ATO'], 'm--', linewidth=4, label='Pareto frontier', alpha=0.5)
        ax2.set_xlabel(r'$R(\tilde \theta_n^\lambda,\hat{P}_{\rm{test}})$')
        ax2.set_ylabel(r'$U(\tilde \theta_n^\lambda,\hat{P}_{\rm{test}})$')
        # ax2.set_title(plot_title)
        ax2.legend(loc='best')

    elif folder == '0a9ae7':
        fig, ax = plt.subplots(1, 1, figsize=set_size(width,fraction=1))
        sns.regplot(x='BCE', y='ATO', data=config_base, fit_reg=False, label=r'Chebyshev $\hat \theta_n^\lambda$',ax=ax,color='c',scatter_kws={'s':6})
        plt.plot(config_pareto['BCE'], config_pareto['ATO'], 'm--', linewidth = 4, label='Pareto frontier', alpha=0.5)
        ax.set_xlabel(r'$R(\tilde \theta_n^\lambda,\hat{P}_{\rm{test}})$')
        ax.set_ylabel(r'$U(\tilde \theta_n^\lambda,\hat{P}_{\rm{test}})$')
        ax.legend(loc='best')
        plt.title(plot_title + ': ' + 'Penalising all internal representations')

    plt.savefig('commit{}_datanum{}.pdf'.format(folder,data_num), format='pdf', bbox_inches='tight')


    # plt.show()


# for data_num, folder in itertools.product(range(0,3), {"e5c0f79"}):
#
#     if data_num == 0:
#         plot_title = 'UCI (gender)'
#     elif data_num == 1:
#         plot_title = 'UCI (race)'
#     elif data_num == 2:  # UCI adult dataset has both gender and race protected attributes
#         plot_title = 'Recidivism (race)'
#
#     inputPoints = []
#     for mc in range(0, 99):
#
#         name = 'compareMethods_data{}_mc{}.p'.format(data_num, mc)
#         filename = path.join(folder, name)
#
#         if path.exists(filename):
#
#             simresults = pickle.load(open(filename, 'rb'))
#             closs = torch.stack(simresults['adv_recorded_sweep_results_test']['c_loss'])
#             causal = torch.stack(simresults['adv_recorded_sweep_results_test']['causal'])
#             closs_causal = torch.stack([closs, causal], dim=1)
#             inputPoints.append(closs_causal.tolist())
#
#     candidatePareto = list(itertools.chain(*inputPoints))
#     candidatePareto_orig = candidatePareto.copy()
#
#     paretoPoints, dominatedPoints = simple_cull(candidatePareto, dominates)
#     print('# of pareto points {}'.format(paretoPoints.__len__()))
#     print('# of dominated points {}'.format(dominatedPoints.__len__()))
#
#     config_base = pd.DataFrame.from_records(candidatePareto_orig,columns=['BCE','ATO'])
#     config_pareto = pd.DataFrame.from_records(paretoPoints,columns=['BCE','ATO']).sort_values('BCE')
#
#     ## Plotting
#
#     # % your document class here
#     # \documentclass[10pt]{report}
#     # \begin{document}
#     #
#     # % gives the width of the current document in pts
#     # \showthe\textwidth
#     #
#     # \end{document}
#
#     width = 397
#
#     plt.style.use('seaborn')
#
#     nice_fonts = {
#             # Use LaTeX to write all text
#             "text.usetex": True,
#             "font.family": "serif",
#             # Use 10pt font in plots, to match 10pt font in document
#             "axes.labelsize": 10,
#             "font.size": 10,
#             # Make the legend/label fonts a little smaller
#             "legend.fontsize": 8,
#             "xtick.labelsize": 8,
#             "ytick.labelsize": 8,
#     }
#
#     mpl.rcParams.update(nice_fonts)
#
#     # plot all the Pareto candidates
#     fig, ax = plt.subplots(1, 1, figsize=set_size(width,fraction=1))
#     sns.regplot(x='BCE', y='ATO', data=config_base, fit_reg=False, label=r'adversarial $\tilde \theta_n^\lambda$',ax=ax)
#
#     # plot the pareto frontier
#     plt.plot(config_pareto['BCE'], config_pareto['ATO'], '--', linewidth = 4, label='Pareto frontier', alpha=0.5)
#     ax.set_xlabel(r'$R(\tilde \theta_n^\lambda,\hat{P}_{\rm{test}})$')
#     ax.set_ylabel(r'$U(\tilde \theta_n^\lambda,\hat{P}_{\rm{test}})$')
#
#     # plt.xlim([0.75, 6])
#     plt.legend(loc='best')
#     if folder == 'e5c0f79':
#         plt.title(plot_title)
#     elif folder == '0a9ae7':
#         plt.title(plot_title)
#
#     plt.savefig('commit{}_datanum{}_adversarial.pdf'.format(folder,data_num), format='pdf', bbox_inches='tight')
#     # plt.show()
