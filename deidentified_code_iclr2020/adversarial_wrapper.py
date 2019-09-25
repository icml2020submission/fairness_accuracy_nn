import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from assessfairness import fairness_measures
from debiasing import populate_sweep_results
from dataset_factory import get_xyz_from_dataloader
import visualize_data
import debiasing_networks
from torch_device_settings import TORCH_DEVICE
import copy


def pretrain_classifier(clf, data_loader, optimizer, criterion):
    for x, y, _, _ in data_loader:
        clf.zero_grad()
        p_y,_ = clf(h=x,z=x) # it doesn't matter what z is
        loss = criterion(p_y, y.view_as(p_y))
        loss.backward()
        optimizer.step()
    return clf

class Adversary(nn.Module):

    def __init__(self, n_sensitive, n_hidden=32):
        super(Adversary, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_sensitive),
        )

    def forward(self, x):
        return torch.sigmoid(self.network(x))


def pretrain_adversary(adv, clf, data_loader, optimizer, criterion):

    for x, _, z, _ in data_loader:
        p_y,_ = clf(h=x,z=x) # it doesn't matter what z is
        adv.zero_grad()
        p_z = adv(p_y)
        loss = criterion(p_z.squeeze(), z.squeeze())
        loss.backward()
        optimizer.step()
    return adv

def train(clf, adv, data_loader, clf_criterion, adv_criterion,
          clf_optimizer, adv_optimizer):

    # Train adversary
    for x, y, z,_ in data_loader:
        p_y,_ = clf(h=x,z=x) # it doesn't matter what z is
        adv.zero_grad()
        p_z = adv(p_y)
        adv_loss = adv_criterion(p_z, z)
        adv_loss.backward()
        adv_optimizer.step()

    # Train classifier on single batch
    for x, y, z, _ in data_loader:
        pass
    p_y, _ = clf(h=x, z=x)  # it doesn't matter what z is, we won't use it
    clf.zero_grad()
    p_z = adv(p_y)
    clf_loss = clf_criterion(p_y, y) - adv_criterion(p_z, z) # (adv_criterion(adv(p_y), z) * lambdas).mean()
    clf_loss.backward()
    clf_optimizer.step()

    return clf, adv, clf_loss


def adversarial(train_loader,
                testing_loader,
                nr_of_layers,
                n_hidden,
                dropout_probability,
                n_protected,
                lambdas,
                propensity_train,
                propensity_test):

    print('# training samples:', len(train_loader.sampler))
    print('training batch size: {}'.format(train_loader.batch_size))
    print('# training batches:', np.ceil(len(train_loader.sampler)/train_loader.batch_size))

    x_train, y_train, z_train = get_xyz_from_dataloader(train_loader)
    x_test, y_test, z_test = get_xyz_from_dataloader(testing_loader)

    # train classifier for predicting y from x
    initial_clf = debiasing_networks.Net(nr_of_layers=nr_of_layers,
                                             train_loader=train_loader,
                                             nr_io=n_hidden,
                                             dropout_probability=dropout_probability,
                                             sigma_reg=None).to(TORCH_DEVICE)
    clf = copy.deepcopy(initial_clf)
    print(clf)
    clf_criterion = nn.BCEWithLogitsLoss()
    clf_optimizer = optim.Adam(clf.parameters(),lr=0.001)
    N_CLF_EPOCHS = 2
    for epoch in range(N_CLF_EPOCHS):
        clf = pretrain_classifier(clf, train_loader, clf_optimizer, clf_criterion)

    # train adversary for predicting z from y
    adv = Adversary(n_sensitive=n_protected)
    adv_criterion = nn.BCELoss(weight=lambdas)
    adv_optimizer = optim.Adam(adv.parameters())
    N_ADV_EPOCHS = 5
    for epoch in range(N_ADV_EPOCHS):
        pretrain_adversary(adv, clf, train_loader, adv_optimizer, adv_criterion)

    # joint training
    N_EPOCH_COMBINED = 200

    criterion = nn.BCEWithLogitsLoss(reduction='mean') #only used for purposes of display
    for epoch in range(1, N_EPOCH_COMBINED):

        clf, adv, clf_loss = train(clf, adv, train_loader, clf_criterion, adv_criterion, clf_optimizer, adv_optimizer)

        if epoch%10==0:

            with torch.no_grad():
                clf_pred, _ = clf(h=x_test)

            fairness_test = fairness_measures(clf_pred, y_test, z_test,propensity_test)
            print('Epoch {}: {}; c_loss_test={}'.format(epoch,fairness_test, criterion(clf_pred, y_test)))



    # calculate criterion and mean variance on test set after training is all finished
    with torch.no_grad():
        output_test, _ = clf(h=x_test, z=x_test) # it doesn't matter what z is
        probs_test = torch.sigmoid(output_test)
        output_train, _ = clf(h=x_train, z=x_test)  # it doesn't matter what z is
        probs_train = torch.sigmoid(output_train)

    fairness_measures_train = fairness_measures(probs_train,
                                                y_train,
                                                z_train,
                                                propensity_train)

    fairness_measures_test = fairness_measures(probs_test,
                                               y_test,
                                               z_test,
                                               propensity_test)
    c_loss_train = criterion(output_train, y_train)
    c_loss_test = criterion(output_test, y_test)
    return c_loss_train, c_loss_test, fairness_measures_train, fairness_measures_test

def adversarial_sweep(train_loader,testing_loader,lambdas,propensity_train,propensity_test,nn_parameters,save_base_directory=None):

    n_hidden = nn_parameters['nr_io']
    nr_of_layers = nn_parameters['nr_of_layers']
    dropout_probability = nn_parameters['dropout_probability']
    n_protected = testing_loader.dataset.z.shape[-1]

    sweep_lambdas = []
    for weight in lambdas:
        sweep_lambdas.append(torch.Tensor([weight]*n_protected))


    print('Start sweeping lambda')
    # weighting for the adversary (not sure why this is unequally weighted)
    # if n_protected == 1:
    #     lambdas = torch.Tensor([100])
    # elif n_protected == 2:
    #     # lambdas = torch.Tensor([130, 30]) # needs to be as many as there are protected variables
    #     lambdas = torch.Tensor([100, 100])  # needs to be as many as there are protected variables
    # else:
    #     raise ValueError('Unknown number of protected variables; only defined for 1 and 2 so far')

    sweep_train = dict()
    sweep_test = dict()

    for l in sweep_lambdas:

        print('adversarial weights: {}'.format(l))
        c_loss_train, c_loss_test, fairness_measures_train, fairness_measures_test = \
            adversarial(train_loader,
                        testing_loader,
                        nr_of_layers,
                        n_hidden,
                        dropout_probability,
                        n_protected,
                        l,
                        propensity_train,
                        propensity_test)

        sweep_train, sweep_test = \
            populate_sweep_results(sweep_train,
                           c_loss_train,
                           fairness_measures_train,
                           sweep_test,
                           c_loss_test,
                           fairness_measures_test)

    if save_base_directory is not None:
        visualize_data.plot_sweep_results(proj_loss_weights=lambdas.cpu().numpy(),
                                          sweep_results_train=sweep_train,
                                          sweep_results_test=sweep_test,
                                          key_xaxis='c_loss',
                                          keys_yaxis=['mv_DP', 'mv_EO', 'mv_EOpp', 'causal'],
                                          save_filename="{}/adv_sweep_closs".format(save_base_directory))

    return sweep_train, sweep_test