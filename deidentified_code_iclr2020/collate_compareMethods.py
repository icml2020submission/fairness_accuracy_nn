import pickle
import matplotlib.pyplot as plt
import visualize_data
import os.path
from os import path
import itertools
import sys


def main(folder):

    for data_num, fairness_key, mc in itertools.product(range(0,3), {'mv_EO', 'mv_EOpp', 'mv_DP','causal'}, range(0, 99)):

        name = 'compareMethods_data{}_mc{}.p'.format(data_num, mc)
        filename = path.join(folder[0],name)

        print(filename)

        if path.exists(filename):
            simresults = pickle.load(open(filename, 'rb'))

            ax = visualize_data.plot_sweep_results(proj_loss_weights=simresults['proj_loss_weights'].cpu().numpy(),
                                              sweep_results_train=simresults['recorded_sweep_results_train'],
                                              sweep_results_test=simresults['recorded_sweep_results_test'],
                                              key_xaxis='c_loss',
                                              keys_yaxis=[fairness_key],
                                              save_filename=None)
            plt.title(simresults['dataset_id'])
            # if 'c_loss_KCmassage' in simresults:
            #     ax.scatter(simresults['c_loss_KCmassage'].detach().numpy(), simresults['fairness_KCmassage'][fairness_key], c='b', label='KC massaging test')
            ax.legend()
            plt.savefig('data{}_{}_mc{}_deepdebias.png'.format(data_num, fairness_key, mc))


            # ax = visualize_data.plot_sweep_results(proj_loss_weights=simresults['lambdas'].cpu().numpy(),
            #                                   sweep_results_train=simresults['adv_recorded_sweep_results_train'],
            #                                   sweep_results_test=simresults['adv_recorded_sweep_results_test'],
            #                                   key_xaxis='c_loss',
            #                                   keys_yaxis=[fairness_key],
            #                                   save_filename=None)
            # if 'c_loss_KCmassage' in simresults:
            #     ax.scatter(simresults['c_loss_KCmassage'].detach().numpy(), simresults['fairness_KCmassage'][fairness_key],
            #                c='b', label='KC massaging test')
            # ax.legend()
            # plt.savefig('data{}_{}_mc{}_adv.png'.format(data_num, fairness_key, mc))



    # aggregate individual mcs into gif for a given dataset and fairness key combination
    for data_num, fairness_key in itertools.product(range(0,3), {'mv_EO', 'mv_EOpp', 'mv_DP','causal'}):

        if os.system('which convert') == 0:
            visualize_data.create_animated_gif(filter='data{}_{}_*_deepdebias.png'.format(data_num, fairness_key),
                                                   out_filename='data{}_{}_deepdebias.gif'.format(data_num,fairness_key))
            # visualize_data.create_animated_gif(filter='data{}_{}_*_adv.png'.format(data_num, fairness_key),
            #                                        out_filename='data{}_{}_adv.gif'.format(data_num,fairness_key))

        else:
            print('INFO: convert command not found. On OSX install with ''brew install imagemagick''. If brew is not installed go to: https://brew.sh/')

# Run the actual program
if __name__ == "__main__":
  print(sys.argv)
  main(sys.argv[1:])