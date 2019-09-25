import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import sklearn

import seaborn as sns
from assessfairness import fairness_measures
import model_evaluation

import os

def calculate_graph_limits(display_x_lim,display_y_lim):
    display_x_lim_ind = display_x_lim
    display_x_lim_joint = display_x_lim
    display_y_lim_ind = display_y_lim
    display_y_lim_joint = display_y_lim

    if display_x_lim=='auto':
        display_x_lim_ind=None
        display_x_lim_joint=None
        display_x_lim=None
    elif type(display_x_lim)==tuple:
        display_x_lim_ind = display_x_lim[0]
        display_x_lim_joint = display_x_lim[1]

    if display_y_lim=='auto':
        display_y_lim_ind=None
        display_y_lim_joint=None
        display_y_lim=None
    elif type(display_y_lim)==tuple:
        display_y_lim_ind = display_y_lim[0]
        display_y_lim_joint = display_y_lim[1]

    return display_x_lim, display_x_lim_ind, display_x_lim_joint, display_y_lim, display_y_lim_ind, display_y_lim_joint


def _my_distplot(vals,hist=False,kde=True,kde_kws=None,label=''):

    # TODO: This now uses a fixed bandwith. Check that we really want to do this
    fixed_bw = 0.01

    if kde_kws is None:
        kde_kws = dict()

    kde_kws['bw'] = fixed_bw

    ax = sns.distplot(vals,hist=hist,kde=kde,kde_kws=kde_kws,label=label)

    return ax

def _plot_results_joint(all_vals,all_zs,current_title='',
                        display_x_as_log=False,display_y_as_log=False,display_x_lim=None,display_y_lim=None,zlabel=None,
                        show_figures=None,
                        ground_truth_plot=False,
                        save_base_directory=None,
                        save_base_filename=None,
                        current_epoch=None,
                        plot_in_one_figure=False):

    title_fontsize = 8
    desired_linewidth = 1

    if plot_in_one_figure:
        plt.clf()

    xrange = None
    yrange = None

    nr_of_zs = all_zs.shape[2]

    for nz in range(nr_of_zs):

        if display_y_lim is None:
            no_y_lim = True
        elif (display_y_lim[nz][0] is None) or (display_y_lim[nz][1] is None):
            no_y_lim = True
        else:
            no_y_lim = False

        vals_z0 = all_vals[all_zs[:,:,nz:nz+1] == 0]
        vals_z1 = all_vals[all_zs[:,:,nz:nz+1] == 1]

        if not plot_in_one_figure:
            plt.clf()
        else:
            plt.subplot(1,nr_of_zs,nz+1)

        ax = _my_distplot(vals_z0, hist=False, kde=True,
                          kde_kws={'shade': True, 'linewidth': desired_linewidth, 'linestyle': 'dashed'},
                          label='protected z=0')
        if display_x_as_log:
            ax.set_xscale('log')
        if display_y_as_log:
            ax.set_yscale('log')

        # TODO: maybe do something smarter here again for x limit
        #if display_x_lim is not None:
        #    ax.set_xlim(left=display_x_lim[nz][0], right=display_x_lim[nz][1])
        ax.set_xlim(left=0.0,right=1.0)

        if display_y_lim is not None:
            plt.ylim(display_y_lim[nz][0], display_y_lim[nz][1])
        ax = _my_distplot(vals_z1, hist=False, kde=True,
                          kde_kws={'shade': True, 'linewidth': desired_linewidth, 'linestyle': 'solid'},
                          label='protected z=1')
        if display_x_as_log:
            ax.set_xscale('log')
        if display_y_as_log:
            ax.set_yscale('log')

        # TODO: maybe do something smarter here again for x limit
        #if display_x_lim is not None:
        #    ax.set_xlim(left=display_x_lim[nz][0], right=display_x_lim[nz][1])
        ax.set_xlim(left=0.0,right=1.0)

        if not no_y_lim:
            plt.ylim(display_y_lim[nz][0], display_y_lim[nz][1])

        fontP_x_small = FontProperties()
        fontP_x_small.set_size('x-small')
        plt.legend(loc='upper right',prop=fontP_x_small)

        if no_y_lim:
            # scale it up
            cyrange = ax.get_ylim()
            plt.ylim(cyrange[0],1.5*cyrange[1])

        cxrange = ax.get_xlim()
        cyrange = ax.get_ylim()

        if xrange is None:
            xrange = [cxrange]
        else:
            xrange.append(cxrange)

        if yrange is None:
            yrange = [cyrange]
        else:
            yrange.append(cyrange)

        plt.title(current_title + '; z=' + zlabel[nz], fontsize=title_fontsize)

        if not plot_in_one_figure:
            if save_base_directory is not None:
                if current_epoch is None:
                    if ground_truth_plot:
                        plt.savefig(os.path.join(save_base_directory, '{}_joint_z_{}_ground_truth.png'.format(save_base_filename, zlabel[nz])))
                    else:
                        plt.savefig(os.path.join(save_base_directory, '{}_joint_z_{}.png'.format(save_base_filename, zlabel[nz])))
                else:
                    if ground_truth_plot:
                        plt.savefig(os.path.join(save_base_directory, '{}_joint_z_{}_epoch_{:05d}_ground_truth.png'.format(save_base_filename, zlabel[nz], current_epoch)))
                    else:
                        plt.savefig(os.path.join(save_base_directory, '{}_joint_z_{}_epoch_{:05d}.png'.format(save_base_filename, zlabel[nz], current_epoch)))
            elif show_figures is not None:
                if show_figures:
                    plt.show()

    if plot_in_one_figure:
        if save_base_directory is not None:
            if current_epoch is None:
                if ground_truth_plot:
                    plt.savefig(os.path.join(save_base_directory,'{}_joint_ground_truth.png'.format(save_base_filename)))
                else:
                    plt.savefig(
                        os.path.join(save_base_directory, '{}_joint.png'.format(save_base_filename)))
            else:
                if ground_truth_plot:
                    plt.savefig(os.path.join(save_base_directory,
                                             '{}_joint_epoch_{:05d}_ground_truth.png'.format(save_base_filename,current_epoch)))
                else:
                    plt.savefig(os.path.join(save_base_directory,
                                             '{}_joint_epoch_{:05d}.png'.format(save_base_filename, current_epoch)))
        elif show_figures is not None:
            if show_figures:
                plt.show()

    return xrange,yrange


def _plot_results_individual(all_vals,all_ys,all_zs,val_ranges,using_continuous_y_intervals,current_title='',
                             display_x_as_log=False,display_y_as_log=False,display_x_lim=None,display_y_lim=None,
                             zlabel=None,
                             show_figures=None,
                             ground_truth_plot=False,
                             save_base_directory=None,
                             save_base_filename=None,
                             current_epoch=None,
                             plot_in_one_figure=False):

    if (show_figures is None) and (save_base_directory is None):
        return

    fontP_x_small = FontProperties()
    fontP_x_small.set_size('x-small')

    fontP_small = FontProperties()
    fontP_small.set_size('small')
    title_fontsize = 8
    desired_linewidth = 1

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    nr_of_colors = len(colors)

    nr_of_zs = all_zs.shape[2]

    xrange = None
    yrange = None

    if plot_in_one_figure:
        plt.clf()

    for nz in range(nr_of_zs):

        if display_y_lim is None:
            no_y_lim = True
        elif (display_y_lim[nz][0] is None) or (display_y_lim[nz][1] is None):
            no_y_lim = True
        else:
            no_y_lim = False

        # and now create a plot that also splits it based on the binary output variable (y)
        if not plot_in_one_figure:
            plt.clf()
        else:
            plt.subplot(1,nr_of_zs,nz+1)

        nr_val_ranges = len(val_ranges)

        max_number_of_intervals = 10
        if nr_val_ranges > max_number_of_intervals:
            pool_number = int(np.ceil(float(nr_val_ranges) / float(max_number_of_intervals)))
            print('INFO: Too many value ranges to plot. Subsampling at {}'.format(pool_number))
        else:
            pool_number = 1

        effective_val_ranges = val_ranges[0::pool_number]
        if effective_val_ranges[-1] != val_ranges[-1]:
            effective_val_ranges.append(val_ranges[-1])

        effective_nr_val_ranges = len(effective_val_ranges)

        #print('\nPlotting {} values: '.format(effective_nr_val_ranges), end='')

        for i, v in enumerate(effective_val_ranges):

            #print('#', end='')

            if i == 0:
                current_y_idx = (all_ys <= v)
            else:
                current_y_idx = (all_ys <= v) * (all_ys > effective_val_ranges[i - 1])

            probs_yv_z0 = all_vals[current_y_idx * (all_zs[:,:,nz:nz+1] == 0)]
            probs_yv_z1 = all_vals[current_y_idx * (all_zs[:,:,nz:nz+1] == 1)]

            if using_continuous_y_intervals:
                title_0 = 'y<={:.2f}; protected z=0'.format(v)
                title_1 = 'y<={:.2f}; protected z=1'.format(v)
            else:
                title_0 = 'y={:.2f}; protected z=0'.format(v)
                title_1 = 'y={:.2f}; protected z=1'.format(v)

            # make sure there are enough data-points for plotting
            if len(probs_yv_z0) > 5:
                ax = _my_distplot(probs_yv_z0, hist=False, kde=True,
                                  kde_kws={'shade': True, 'linewidth': desired_linewidth, 'linestyle': 'dashed',
                                           'color': colors[i % nr_of_colors]}, label=title_0)

                if display_x_as_log:
                    ax.set_xscale('log')
                if display_y_as_log:
                    ax.set_yscale('log')

                # TODO: maybe do something smarter here again for x limit
                #if display_x_lim is not None:
                #    ax.set_xlim(left=display_x_lim[nz][0], right=display_x_lim[nz][1])
                ax.set_xlim(left=0.0,right=1.0)

                if not no_y_lim:
                    plt.ylim(display_y_lim[nz][0],display_y_lim[nz][1])

            if len(probs_yv_z1) > 5:
                ax = _my_distplot(probs_yv_z1, hist=False, kde=True,
                                  kde_kws={'shade': True, 'linewidth': desired_linewidth, 'linestyle': 'solid',
                                           'color': colors[i % nr_of_colors]}, label=title_1)

                if display_x_as_log:
                    ax.set_xscale('log')
                if display_y_as_log:
                    ax.set_yscale('log')

                # TODO: maybe do something smarter here again for x limit
                #if display_x_lim is not None:
                #    ax.set_xlim(left=display_x_lim[nz][0], right=display_x_lim[nz][1])
                ax.set_xlim(left=0.0,right=1.0)

                if not no_y_lim:
                    plt.ylim(display_y_lim[nz][0],display_y_lim[nz][1])

        plt.legend(loc='upper right', prop=fontP_x_small)
        # plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", prop=fontP_x_small)
        # plt.legend(loc='upper right')

        plt.title(current_title + '; z=' + zlabel[nz], fontsize=title_fontsize)

        if no_y_lim:
            # scale it up
            cyrange = ax.get_ylim()
            plt.ylim(cyrange[0],1.2*cyrange[1])

        cxrange = ax.get_xlim()
        cyrange = ax.get_ylim()

        if xrange is None:
            xrange = [cxrange]
        else:
            xrange.append(cxrange)
        if yrange is None:
            yrange = [cyrange]
        else:
            yrange.append(cyrange)

        if not plot_in_one_figure:
            if save_base_directory is not None:
                if current_epoch is None:
                    if ground_truth_plot:
                        plt.savefig(os.path.join(save_base_directory, '{}_individual_z_{}_ground_truth.png'.format(save_base_filename,zlabel[nz])))
                    else:
                        plt.savefig(os.path.join(save_base_directory, '{}_individual_z_{}.png'.format(save_base_filename,zlabel[nz])))
                else:
                    if ground_truth_plot:
                        plt.savefig(os.path.join(save_base_directory,'{}_individual_z_{}_epoch_{:05d}_ground_truth.png'.format(save_base_filename,zlabel[nz],current_epoch)))
                    else:
                        plt.savefig(os.path.join(save_base_directory,'{}_individual_z_{}_epoch_{:05d}.png'.format(save_base_filename, zlabel[nz],current_epoch)))

            elif show_figures is not None:
                if show_figures:
                    plt.show()

    if plot_in_one_figure:
        if save_base_directory is not None:
            if current_epoch is None:
                if ground_truth_plot:
                    plt.savefig(os.path.join(save_base_directory,
                                             '{}_individual_ground_truth.png'.format(save_base_filename)))
                else:
                    plt.savefig(os.path.join(save_base_directory,
                                             '{}_individual.png'.format(save_base_filename)))
            else:
                if ground_truth_plot:
                    plt.savefig(os.path.join(save_base_directory,
                                             '{}_individual_epoch_{:05d}_ground_truth.png'.format(
                                                 save_base_filename, current_epoch)))
                else:
                    plt.savefig(os.path.join(save_base_directory,
                                             '{}_individual_epoch_{:05d}.png'.format(save_base_filename, current_epoch)))

        elif show_figures is not None:
            if show_figures:
                plt.show()

    return xrange,yrange

# TODO: clean this up! give it a better name! add some documentation!
def do_testing(net,criterion,testing_loader,proj_loss_weight,
               c_loss_global_multiplier,
               proj_loss_global_multiplier,turn_protected_projection_on=False,title_prefix=None,
               display_x_as_log=False,display_y_as_log=False, display_x_lim=None, display_y_lim=None,
               save_base_filename=None,save_base_directory=None,show_figures=True,
               xlabel=None,ylabel=None,zlabel=None,current_epoch=None):


    display_x_lim_ind = display_x_lim
    display_x_lim_joint = display_x_lim
    display_y_lim_ind = display_y_lim
    display_y_lim_joint = display_y_lim

    if display_x_lim=='auto':
        display_x_lim_ind=None
        display_x_lim_joint=None
        display_x_lim=None
    elif type(display_x_lim)==tuple:
        display_x_lim_ind = display_x_lim[0]
        display_x_lim_joint = display_x_lim[1]

    if display_y_lim=='auto':
        display_y_lim_ind=None
        display_y_lim_joint=None
        display_y_lim=None
    elif type(display_y_lim)==tuple:
        display_y_lim_ind = display_y_lim[0]
        display_y_lim_joint = display_y_lim[1]

    val_ranges = testing_loader.dataset.get_y_val_ranges()
    using_continuous_y_intervals = testing_loader.dataset.continuous_y()

    all_probs, all_output_vals,all_ys, all_zs, running_criterion_loss, _ = \
        model_evaluation.evaluate_net_testing_loader(net=net,
                                          testing_loader=testing_loader,
                                          criterion=criterion)
    testing_accuracy = -1 #TODO: for backward compatibility, remove in future since we are not thresholding

    if not using_continuous_y_intervals:
        all_yhats = (all_probs>=0.5).astype('float32')

        nr_testing = len(testing_loader.sampler)
        baseline_accuracy = model_evaluation.compute_baseline_accuracy(all_ys)

        fairness = fairness_measures(all_probs, all_ys, all_zs, propensity)

    else:
        # TODO: [S] implement fairness measures for continuous response, an easy hack is to bin the continuous response and use response above
        fairness = dict()


    all_vals = None
    if all_probs is not None:
        all_vals = all_probs
    elif all_output_vals is not None:
        all_vals = all_output_vals
    else:
        raise ValueError('Either probs or output values should be given')

    #ks2 = scipy.stats.ks_2samp(probs_z0.reshape(-1), probs_z1.reshape(-1))

    if current_epoch is not None:
        current_title = 'Epoch: {}: '.format(current_epoch)
    else:
        current_title = ''

    # current_title += 'testing dist(probhat): testing acc={:.3f}'.format(testing_accuracy)
    current_title += 'testing dist(probhat):'
    if title_prefix is not None:
        current_title = title_prefix + ': ' + current_title
    current_title += '; c_loss={:.3f}'.format(running_criterion_loss)
    # current_title += '; p_loss={:.3f}\n'.format(running_proj_loss)
    if net.use_batch_normalization:
        current_title += 'BN'
    if net.use_protected_projection:
        current_title += '; PP'
        if net.sigma_reg > 0:
            current_title += '; sigma_reg={:.3f}'.format(net.sigma_reg)
        current_title += '; p_penalty={:.3f}'.format(proj_loss_weight)

    # now do all the fairness measures
    current_count = 0
    for i, (k,v) in enumerate(fairness.items()):
        if k=='two sample KS stat':
            current_title += '\n {}: stat={:.3f}; p={:.3f}'.format(k,fairness[k][0],fairness[k][1])
            current_count = 0
        else:
            if current_count%2==0:
                current_title += '\n'
            else:
                current_title += '; '
            if type(v) == type(''):
                current_title += '{}={}'.format(k,v)
            else:
                current_title += '{}={:.3f}'.format(k,v)
            current_count+=1

    # first do the plot just based on the protected variable
    if (current_epoch is None) and show_figures:
        will_plot = True
    elif show_figures:
        will_plot = True
    elif save_base_filename is not None:
        will_plot = True
    else:
        will_plot = False

    if will_plot:
        print('\nStarting plotting')

    # plot for the true data

    current_title_true = 'Ground truth/gold standard data'

    if (current_epoch is None) and show_figures:
        # ground truth is not epoch dependent, so let's not print it all the time
        _, _ = _plot_results_joint(all_vals=all_ys, all_zs=all_zs, current_title=current_title_true,
                                   display_x_as_log=display_x_as_log,
                                   display_y_as_log=display_y_as_log,
                                   display_x_lim=display_x_lim_joint,
                                   display_y_lim=display_y_lim_joint,
                                   zlabel=zlabel,
                                   ground_truth_plot=True,
                                   show_figures=show_figures,
                                   current_epoch=current_epoch)

    if show_figures:
        xrange_ind, yrange_ind = _plot_results_individual(all_vals=all_vals, all_ys=all_ys, all_zs=all_zs,
                                                          val_ranges=val_ranges,
                                                          using_continuous_y_intervals=using_continuous_y_intervals,
                                                          current_title=current_title,
                                                          display_x_as_log=display_x_as_log,
                                                          display_y_as_log=display_y_as_log,
                                                          display_x_lim=display_x_lim_ind, display_y_lim=display_y_lim_ind,
                                                          zlabel=zlabel,
                                                          show_figures=show_figures,
                                                          current_epoch=current_epoch)

        xrange_joint, yrange_joint=_plot_results_joint(all_vals=all_vals, all_zs=all_zs, current_title=current_title,
                                                       display_x_as_log=display_x_as_log, display_y_as_log=display_y_as_log, display_x_lim=display_x_lim_joint, display_y_lim=display_y_lim_joint,
                                                       zlabel=zlabel,
                                                       show_figures=show_figures,
                                                       current_epoch=current_epoch)
    else:
        xrange_ind = (0.0,1.0)
        xrange_joint = (0.0,1.0)
        yrange_ind = (None,None)
        yrange_joint = (None,None)



    if save_base_filename is not None:

        if save_base_directory is None:
            save_base_directory = '.'
        else:
            if not os.path.exists(save_base_directory):
                os.mkdir(save_base_directory)

        # print it

        _, _ = _plot_results_individual(all_vals=all_vals, all_ys=all_ys, all_zs=all_zs,
                                        val_ranges=val_ranges,
                                        using_continuous_y_intervals=using_continuous_y_intervals,
                                        current_title=current_title,
                                        display_x_as_log=display_x_as_log,
                                        display_y_as_log=display_y_as_log,
                                        display_x_lim=display_x_lim_ind, display_y_lim=display_y_lim_ind,
                                        zlabel=zlabel,
                                        save_base_directory=save_base_directory,
                                        save_base_filename=save_base_filename,
                                        current_epoch=current_epoch)

        _, _ = _plot_results_joint(all_vals=all_vals, all_zs=all_zs, current_title=current_title,
                                   display_x_as_log=display_x_as_log, display_y_as_log=display_y_as_log,
                                   display_x_lim=display_x_lim_joint, display_y_lim=display_y_lim_joint,
                                   zlabel=zlabel,
                                   save_base_directory=save_base_directory,
                                   save_base_filename=save_base_filename,
                                   current_epoch=current_epoch)

        if current_epoch is None:
            # ground truth is not epoch dependent, so let's not print it all the time
            _, _ = _plot_results_joint(all_vals=all_ys, all_zs=all_zs, current_title=current_title_true,
                                       display_x_as_log=display_x_as_log,
                                       display_y_as_log=display_y_as_log,
                                       display_x_lim=display_x_lim_joint,
                                       display_y_lim=display_y_lim_joint,
                                       zlabel=zlabel,
                                       ground_truth_plot=True,
                                       save_base_directory=save_base_directory,
                                       save_base_filename=save_base_filename,
                                       current_epoch=current_epoch)

    if will_plot:
        print('\nEnd plotting\n')

    xrange = (xrange_ind,xrange_joint)
    yrange = (yrange_ind,yrange_joint)

    # return fairness_measures, testing_accuracy, all_probs, all_output_vals,all_ys, all_zs, running_loss, running_criterion_loss, xrange, yrange

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)