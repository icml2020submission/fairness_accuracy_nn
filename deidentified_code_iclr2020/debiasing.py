import copy
import os
import random
import numpy as np
import torch
from torch_device_settings import TORCH_DEVICE
import torch.optim as optim
import torch.nn as nn

import temperature_scaling


# import all our own modules
import debiasing_networks
import dataset_factory
import model_testing
import model_evaluation
import custom_lr_scheduler
import visualize_data
from assessfairness import fairness_measures
from matplotlib import pyplot as plt

def populate_sweep_results(recorded_sweep_results_train,
                           c_loss_train,
                           fairness_measures_train,
                           recorded_sweep_results_test,
                           c_loss_test,
                           fairness_measures_test):

    # populate sweep results for TRAINING set

    if 'c_loss' in recorded_sweep_results_train:
        recorded_sweep_results_train['c_loss'].append(c_loss_train)
    else:
        recorded_sweep_results_train['c_loss'] = [c_loss_train]

    for k in fairness_measures_train:

        if k in recorded_sweep_results_train:
            recorded_sweep_results_train[k].append(fairness_measures_train[k])
        else:
            recorded_sweep_results_train[k] = [fairness_measures_train[k]]

    # populate sweep results for TEST set

    if 'c_loss' in recorded_sweep_results_test:
        recorded_sweep_results_test['c_loss'].append(c_loss_test)
    else:
        recorded_sweep_results_test['c_loss'] = [c_loss_test]

    for k in fairness_measures_test:
        if k in recorded_sweep_results_test:
            recorded_sweep_results_test[k].append(fairness_measures_test[k])
        else:
            recorded_sweep_results_test[k] = [fairness_measures_test[k]]

    return recorded_sweep_results_train, recorded_sweep_results_test


def load_data_for_debiasing_sweep(dataset_id='S_S',
                                  number_of_continuous_y_intervals=None,
                                  desired_seed=None,
                                  desired_number_of_samples=5000,
                                  show_data=True,
                                  debias_type=None,
                                  testing_percent=50,
                                  desired_batch_size=None):

    if desired_seed is not None:
        print('Setting the random seed to {:}'.format(desired_seed))
        random.seed(desired_seed)
        np.random.seed(desired_seed)
        torch.manual_seed(desired_seed)

    # load the data
    # read or create the data
    dataset, x, y, z, xl, yl, zl, use_continuous_y = \
        dataset_factory.get_dataset_by_id(id=dataset_id,
                                          debias_type=debias_type,
                                          number_of_continuous_y_intervals=number_of_continuous_y_intervals,
                                          desired_number_of_samples=desired_number_of_samples,
                                          visualize_data=show_data)

    # do a train test split
    train_loader,testing_loader = \
        dataset_factory.get_data_loaders(dataset=dataset,x=x,y=y,z=z,
                                         desired_batch_size=desired_batch_size,
                                         testing_percent=testing_percent)

    output = dict()
    output['xl'] = xl
    output['yl'] = yl
    output['zl'] = zl
    output['use_continuous_y'] = use_continuous_y
    output['train_loader'] = train_loader
    output['testing_loader'] = testing_loader

    return output


def debiasing_sweep(debiasing_obj,
                    desired_seed,
                    proj_loss_weights,
                    save_figures=None,
                    save_base_directory=None,
                    show_figures=False):

    if desired_seed is not None:
        print('Setting the random seed to {:}'.format(desired_seed))
        random.seed(desired_seed)
        np.random.seed(desired_seed)
        torch.manual_seed(desired_seed)

    #TODO: eventually implement smarter way to pick spread of weights, using NBI of DasDennis2000?
    if proj_loss_weights is None:
        proj_loss_weights = torch.log(torch.linspace(1.0,np.exp(1.0),15))
        print('INFO: set default projection loss weights to {}'.format(proj_loss_weights))

    for proj_loss_weight in proj_loss_weights:
        if (proj_loss_weight < 0) or (proj_loss_weight > 1.0):
            print('\n\nWARNING: Projection loss weight should normally be in [0,1], but was set to {:.3f}.\n\n'.format(proj_loss_weight))

    # sort the energy values
    proj_loss_weights = proj_loss_weights.sort()[0]

    recorded_sweep_results_train = dict()
    recorded_sweep_results_test = dict()
    phats_list = list()



    # for zero:
    c_loss_train0, \
    c_loss_test0, \
    fairness_measures_train_0, \
    fairness_measures_test_0, \
    baseline_accuracy_0, \
    min_loss0, \
    max_loss0, \
    phats0 = debiasing_obj.main(proj_loss_weight=0.0,
                                c_loss_global_multiplier=[1.0, 0.0],
                                proj_loss_global_multiplier=[1.0, 0.0])

    recorded_sweep_results_train, recorded_sweep_results_test = \
        populate_sweep_results(recorded_sweep_results_train,
                               c_loss_train0,
                               fairness_measures_train_0,
                               recorded_sweep_results_test,
                               c_loss_test0,
                               fairness_measures_test_0)

    # for one:
    c_loss_train_1,\
    c_loss_test_1, \
    fairness_measures_train_1, \
    fairness_measures_test_1, \
    baseline_accuracy_1, \
    min_loss1, \
    max_loss1,\
    phats1 = debiasing_obj.main(proj_loss_weight=1.0,
                                c_loss_global_multiplier=[1.0, 0.0],
                                proj_loss_global_multiplier=[1.0, 0.0])



    # this was the previous auto-normalize feature, which we always perform now
    # each loss is standardized by (f_i - f_i^min)/(f_i^max - f_i^min)
    proj_loss_global_multiplier = [1./(max_loss1 - min_loss1), min_loss1]
    c_loss_global_multiplier = [1./(max_loss0 - min_loss0), min_loss0]
    print('Autotuned proj_loss_global_multiplier = {}'.format(proj_loss_global_multiplier))
    print('Autotuned c_loss_global_multiplier = {}'.format(c_loss_global_multiplier))

    print('Start sweeping lambda')

    for proj_loss_weight in proj_loss_weights[1:-1]:


        c_loss_train, \
        c_loss_test, \
        fairness_measures_train, \
        fairness_measures_test, \
        baseline_accuracy, \
        _, _,\
        phats\
            = debiasing_obj.main(proj_loss_weight=proj_loss_weight,
                                 c_loss_global_multiplier=c_loss_global_multiplier,
                                 proj_loss_global_multiplier=proj_loss_global_multiplier)

        recorded_sweep_results_train, recorded_sweep_results_test = \
            populate_sweep_results(recorded_sweep_results_train,
                           c_loss_train,
                           fairness_measures_train,
                           recorded_sweep_results_test,
                           c_loss_test,
                           fairness_measures_test)

        phats_list.append(phats)

    recorded_sweep_results_train, recorded_sweep_results_test = \
        populate_sweep_results(recorded_sweep_results_train,
                               c_loss_train_1,
                               fairness_measures_train_1,
                               recorded_sweep_results_test,
                               c_loss_test_1,
                               fairness_measures_test_1)


    print('Finished sweeping lambda')

    if show_figures or save_figures:

        if not os.path.exists(save_base_directory):
            print('INFO: Creating directory: {}'.format(save_base_directory))
            os.mkdir(save_base_directory)

        visualize_data.plot_sweep_results(proj_loss_weights=proj_loss_weights.cpu().numpy(),
                                          sweep_results_train=recorded_sweep_results_train,
                                          sweep_results_test=recorded_sweep_results_test,
                                          key_xaxis='c_loss',
                                          keys_yaxis=['mv_DP', 'mv_EO', 'mv_EOpp','causal'],
                                          save_filename="{}/sweep_closs".format(save_base_directory))


    # create animated gif over the different weight settings
    # if create_animated_gif:
    #     if os.system('which convert') == 0:
    #         for zn in zl:
    #             visualize_data.create_animated_gif(filter='{}/*_individual_z_{}.png'.format(save_base_directory, zn),
    #                                                out_filename='all_individual_z_{}.gif'.format(zn))
    #             visualize_data.create_animated_gif(filter='{}/*_joint_z_{}.png'.format(save_base_directory, zn),
    #                                                out_filename='all_joint_z_{}.gif'.format(zn))
    #     else:
    #         print(
    #             'INFO: convert command not found. On OSX install with ''brew install imagemagick''. If brew is not installed go to: https://brew.sh/')

    return recorded_sweep_results_train, recorded_sweep_results_test, phats_list


class Debiasing():
    def __init__(self,
                 xl,
                 yl,
                 zl,
                 use_continuous_y,
                 train_loader,
                 testing_loader,
                 nr_of_layers=4,
                 nr_io=2,
                 debias_type='causal_inference',
                 debias_individual_layers_index=None,
                 penalize_fit=False,
                 scheduler_factor=None,
                 scheduler_patience=None,
                 dropout_probability=0.2,
                 desired_number_of_epochs=500,
                 save_figures=False,
                 show_figures=False,
                 use_protected_projection=True,
                 use_only_projection_penalty=True,
                 active_unit='relu',
                 use_batch_normalization=False,
                 learning_rate=0.001,
                 use_testing_loss_for_scheduler=False,
                 create_animated_gif=False,
                 save_base_directory=None,
                 figure_iter_output=None,
                 scalarization='chebychev'
                 ):

        self.xl = xl
        self.yl = yl
        self.zl = zl
        self.use_continuous_y = use_continuous_y
        self.train_loader = train_loader
        self.testing_loader = testing_loader

        self.nr_of_layers = nr_of_layers
        self.nr_io = nr_io
        self.dropout_probability = dropout_probability
        self.debias_type = debias_type
        self.debias_individual_layers_index = debias_individual_layers_index
        self.penalize_fit = penalize_fit
        self.use_protected_projection = use_protected_projection
        self.use_only_projection_penalty = use_only_projection_penalty
        self.active_unit = active_unit
        self.use_batch_normalization = use_batch_normalization

        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.learning_rate = learning_rate
        self.use_testing_loss_for_scheduler = use_testing_loss_for_scheduler

        self.scalarization = scalarization

        self.desired_number_of_epochs = desired_number_of_epochs

        self.save_figures = save_figures
        self.show_figures = show_figures

        self.create_animated_gif = create_animated_gif
        self.save_base_directory = save_base_directory
        self.figure_iter_output = figure_iter_output

        propnet, temp, propensity_train, propensity_test = self.propensity()
        self.propnet = propnet
        self.temp = temp
        self.propensity_train = propensity_train
        self.propensity_test = propensity_test

        x_train, y_train, z_train = dataset_factory.get_xyz_from_dataloader(self.train_loader)
        x_test, y_test, z_test = dataset_factory.get_xyz_from_dataloader(self.testing_loader)
        self.x_train = x_train
        self.y_train = y_train
        self.z_train = z_train
        self.x_test = x_test
        self.y_test = y_test
        self.z_test = z_test

    def propensity(self):

        x_train, y_train, z_train = dataset_factory.get_xyz_from_dataloader(self.train_loader)
        x_test, y_test, z_test = dataset_factory.get_xyz_from_dataloader(self.testing_loader)

        if self.debias_type == 'causal_inference':

            N_propnet_EPOCHS = 100

            nr_inputs = self.train_loader.dataset.x.size()[2]
            propnet = propensityNet(n_features=nr_inputs)
            propnet_optimizer = optim.Adam(propnet.parameters())
            print('Begin training propensity score network')
            for epoch in range(N_propnet_EPOCHS):
                propnet.train()
                propnet = train_propnet(epoch, propnet, self.train_loader, x_test, z_test, propnet_optimizer)
            print('Finished training propensity score network')

            # TODO: should the propensity net be trained again for test set? Right now this option is turned off
            # setz = np.unique(testing_loader.dataset.z, axis=0)
            # setz = torch.from_numpy(setz)
            # propnet_test = propensityNet(n_features=nr_inputs, n_target=setz.shape[0])
            # propnet_test_optimizer = optim.Adam(propnet_test.parameters())
            # for epoch in range(N_propnet_EPOCHS):
            #     propnet_test = train_propensityNet(propnet_test, testing_loader, propnet_test_optimizer, setz)

            # calibrate according to temperature_scaling paper Guo, Pleiss, et. al
            propnet_train_calibrated = temperature_scaling.ModelWithTemperature(propnet)
            propnet_train_calibrated.set_temperature(self.train_loader)
            temp = propnet_train_calibrated.state_dict()['temperature']

            with torch.no_grad():
                propensity_train = nn.functional.softmax(propnet(x_train)/temp, dim=2)
                propensity_test = nn.functional.softmax(propnet(x_test)/temp, dim=2)

        else:
            propnet, temp, propensity_train, propensity_test = None, None, None, None

        return propnet, temp, propensity_train, propensity_test

    def main(self, proj_loss_weight, c_loss_global_multiplier, proj_loss_global_multiplier):

        # HERE GO THE SETTINGS TO PLAY WITH

        # general settings

        # for LDA the covariance matrix is regularized via sigma_reg; set to zero if you do not want regularization
        sigma_reg = 0.0

        display_interval = self.desired_number_of_epochs//10        # how often output should be displayed
        display_x_as_log = False    # log transform x axis for plotting
        display_y_as_log = False    # log transform y axis for plotting
        display_x_lim = 'auto' #None #[-5,25]   # set to None if this should not be limited in display; set to 'auto' for automatic mode (consistent across sweeps)
        display_y_lim = 'auto'

        # SETTINGS TO PLAY WITH END HERE

        nr_training = self.x_train.shape[0]

        if not self.use_continuous_y:
            # compute baseline accuracy
            baseline_accuracy = model_evaluation.compute_baseline_accuracy(self.train_loader.dataset.y)
            print('Baseline accuracy = {:.2f}'.format(baseline_accuracy))
        else:
            baseline_accuracy = None

        #criterion = nn.CrossEntropyLoss() # do this for multi-label
        if self.use_continuous_y:
            print('INFO: Using MSE loss')
            criterion = nn.MSELoss(reduction='sum')
        else:
            # TODO: support multi-class y
            print('INFO: Using binary cross entropy loss')
            criterion = nn.BCEWithLogitsLoss(reduction='mean')


        initial_network = debiasing_networks.Net(nr_of_layers=self.nr_of_layers,
                                                 train_loader=self.train_loader,
                                                 nr_io=self.nr_io,
                                                 debias_type=self.debias_type,
                                                 active_unit=self.active_unit,
                                                 penalize_fit=self.penalize_fit,
                                                 dropout_probability=self.dropout_probability,
                                                 use_batch_normalization=self.use_batch_normalization,
                                                 use_protected_projection=self.use_protected_projection,
                                                 use_only_projection_penalty=self.use_only_projection_penalty,
                                                 sigma_reg=sigma_reg,
                                                 debias_individual_layers_index=self.debias_individual_layers_index).to(TORCH_DEVICE)



        # go back to the original initialization
        net = copy.deepcopy(initial_network)
        print(net)

        print('\nStarting training for proj_loss_weight={:.3f}'.format(proj_loss_weight))

        # create an optimizer
        optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)

        # create a step-size scheduler
        scheduler = custom_lr_scheduler.CustomReduceLROnPlateau(optimizer, 'min', verbose=True, factor=self.scheduler_factor, patience=self.scheduler_patience, eps=1e-6)

        # figure output
        if self.save_figures:
            if not os.path.exists(self.save_base_directory):
                print('INFO: Creating directory: {}'.format(self.save_base_directory))
                os.mkdir(self.save_base_directory)
            current_save_base_filename = 'proj_loss_weight_{:.3f}'.format(proj_loss_weight)
        else:
            current_save_base_filename = None

        losses = []

        for epoch in range(self.desired_number_of_epochs):  # loop over the dataset multiple times

            running_loss = 0.0

            net.train()

            for i, data in enumerate(self.train_loader, 0):

                # get the inputs
                x_sample, y_sample, z_sample, group_indices = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs, proj_loss = net(h=x_sample,
                                         z=z_sample,
                                         y=y_sample,
                                         turn_protected_projection_on=True,
                                         current_epoch=epoch,
                                         group_indices=group_indices,
                                         propnet=self.propnet,
                                         propnet_temp=self.temp)

                c_loss = criterion(outputs, y_sample.view_as(outputs)) # criterion reduction is mean
                if self.scalarization == 'convexcombo':
                    loss = (1.0 - proj_loss_weight) * (c_loss - c_loss_global_multiplier[1]) * c_loss_global_multiplier[0] \
                           + proj_loss_weight * (proj_loss - proj_loss_global_multiplier[1]) * proj_loss_global_multiplier[0]
                elif self.scalarization == 'chebychev':
                    loss = torch.max(
                        (1.0 - proj_loss_weight) * (c_loss - c_loss_global_multiplier[1]) * c_loss_global_multiplier[0],
                        proj_loss_weight * (proj_loss - proj_loss_global_multiplier[1]) * proj_loss_global_multiplier[0])
                # running_loss is be used for scheduler
                running_loss += loss*x_sample.shape[0]/nr_training

                loss.backward()
                optimizer.step()

                losses.append(loss.detach())


            if epoch % display_interval == 0:

                net.eval()
                with torch.no_grad():
                    output_train,_ = net(h=self.x_train, turn_protected_projection_on=False)
                    probs_train = torch.sigmoid(output_train)
                    output_test,_ = net(h=self.x_test, turn_protected_projection_on=False)
                    probs_test = torch.sigmoid(output_test)

                    c_loss_train = criterion(output_train, self.y_train)
                    c_loss_test = criterion(output_test, self.y_test)

                    fairness_measures_train = fairness_measures(probs_train,
                                                                self.y_train,
                                                                self.z_train,
                                                                self.propensity_train)

                    fairness_measures_test = fairness_measures(probs_test,
                                                               self.y_test,
                                                               self.z_test,
                                                               self.propensity_test)



                    print('Epoch {}: c_loss_train = {:.3f}; fair_train = {} '
                          'c_loss_test = {:.3f}; fair_test = {}'
                          .format(epoch, c_loss_train, fairness_measures_train,
                                  c_loss_test, fairness_measures_test))


            if self.figure_iter_output is not None:
                if (epoch % self.figure_iter_output == 0) or (epoch == self.desired_number_of_epochs-1):
                    model_testing.do_testing(net=net,
                                             criterion=criterion,
                                             testing_loader=self.testing_loader,
                                             c_loss_global_multiplier=c_loss_global_multiplier,
                                             proj_loss_weight=proj_loss_weight,
                                             proj_loss_global_multiplier=proj_loss_global_multiplier,
                                             turn_protected_projection_on=turn_protected_projection_on,
                                             display_x_as_log=display_x_as_log,
                                             display_y_as_log=display_y_as_log,
                                             display_x_lim=display_x_lim,
                                             display_y_lim=display_y_lim,
                                             save_base_filename=current_save_base_filename,
                                             save_base_directory=self.save_base_directory,
                                             show_figures=self.show_figures,
                                             xlabel=self.xl,
                                             ylabel=self.yl,
                                             zlabel=self.zl,
                                             current_epoch=epoch)

            # between epochs
            if self.use_testing_loss_for_scheduler:
                # recall that proj_loss_test doesn't actually mean anything, cheat by setting testing_loss to convex combination of c_loss_test and fairness_measures_test
                # TODO: avoid cheating by using validation set
                testing_loss = (1.0 - proj_loss_weight) * c_loss_test + proj_loss_weight * fairness_measures_test['causal']
                scheduler.step(testing_loss.item())
            else:
                scheduler.step(running_loss.item())

            if scheduler.has_convergence_been_reached():
                print('INFO: Converence has been reached. Stopping iterations.')
                break


        print('Finished training')
        min_loss = min(losses)
        max_loss = max(losses)

        net.eval()
        with torch.no_grad():
            output_train, _ = net(h=self.x_train, turn_protected_projection_on=False)
            probs_train = torch.sigmoid(output_train)
            output_test, _ = net(h=self.x_test, turn_protected_projection_on=False)
            probs_test = torch.sigmoid(output_test)


            fairness_measures_train = fairness_measures(probs_train,
                                                        self.y_train,
                                                        self.z_train,
                                                        self.propensity_train)

            fairness_measures_test = fairness_measures(probs_test,
                                                       self.y_test,
                                                       self.z_test,
                                                       self.propensity_test)


            c_loss_train = criterion(output_train, self.y_train)
            c_loss_test = criterion(output_test, self.y_test)


        if self.figure_iter_output is not None:
            # create animated gifs over the epochs
            if self.create_animated_gif:

                # plot individual and joint phat distributions
                # TODO: this doesn't work if testing_loader has minibatches?
                model_testing.do_testing(net=net,
                                         criterion=criterion,
                                         testing_loader=self.testing_loader,
                                         c_loss_global_multiplier=c_loss_global_multiplier,
                                         proj_loss_weight=proj_loss_weight,
                                         proj_loss_global_multiplier=proj_loss_global_multiplier,
                                         turn_protected_projection_on=turn_protected_projection_on,
                                         display_x_as_log=display_x_as_log,
                                         display_y_as_log=display_y_as_log,
                                         display_x_lim=display_x_lim,
                                         display_y_lim=display_y_lim,
                                         save_base_filename=current_save_base_filename,
                                         save_base_directory=self.save_base_directory,
                                         show_figures=self.show_figures,
                                         xlabel=self.xl,
                                         ylable=self.yl,
                                         zlabel=self.zl)

                if os.system('which convert') == 0:
                    for zn in self.zl:
                        visualize_data.create_animated_gif(filter='{}/{}_individual_z_{}_epoch_*.png'.format(self.save_base_directory, current_save_base_filename, zn),
                                                           out_filename='all_individual_z_{}_w_{:.3f}_over_epochs.gif'.format(zn,proj_loss_weight))
                        visualize_data.create_animated_gif(filter='{}/{}_joint_z_{}_epoch_*.png'.format(self.save_base_directory, current_save_base_filename, zn),
                                                           out_filename='all_joint_z_{}_w_{:.3f}_over_epochs.gif'.format(zn,proj_loss_weight))
                else:
                    print('INFO: convert command not found. On OSX install with ''brew install imagemagick''. If brew is not installed go to: https://brew.sh/')


        return c_loss_train, \
               c_loss_test, \
               fairness_measures_train, \
               fairness_measures_test, \
               baseline_accuracy, \
               min_loss, \
               max_loss, \
               probs_test


class propensityNet(nn.Module):

    def __init__(self, n_features, n_hidden=32, p_dropout=0.2):
        """
        Predict P(Z | X) for binary Z
        :param n_features: dim of X
        :param n_hidden:
        :param p_dropout:
        return: raw, unnormalized prediction of P(Z|X), N by 2
        """
        super(propensityNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden, 2),
        )

    def forward(self, x):
        return self.network(x)


def train_propnet(epoch, propnet, data_loader, x_test, z_test, optimizer):
    """

    :param propnet: propensity network to be trained for one epoch
    :param data_loader:
    :param optimizer:
    :return:
    """
    for x, _, z, _ in data_loader:

        propnet.zero_grad()
        predz = propnet(x) # predz[:,:,0] is unnormalized score for z=0, predz[:,:,1] is unnormalized score for z=1
        criterion = nn.CrossEntropyLoss()
        loss = criterion(predz[:,0,:],z[:,0,0].long())
        # TODO: assess loss of propensity score estimation by assessing covariance balance

        loss.backward()
        optimizer.step()


    propnet.eval()
    with torch.no_grad():
        predz_test = propnet(x_test)
        loss_test = criterion(predz_test[:,0,:],z_test[:,0,0].long())

    print('Epoch {}: celoss {}; celoss_test {}'.format(epoch, loss, loss_test))

    return propnet