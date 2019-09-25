import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import pickle
import copy
import sys

# import all our own modules
import debiasing_networks
from assessfairness import fairness_measures
from KamiranCalders2012 import Kamiran_wrapper
import debiasing
import custom_lr_scheduler
from adversarial_wrapper import adversarial_sweep
from dataset_factory import get_xyz_from_dataloader, get_data_loader_from_given_data


def get_kamiran_results(dict_load_data, desired_batch_size, nn_parameters, propensity_test):

    x_train, y_train, z_train = get_xyz_from_dataloader(dict_load_data['train_loader'])
    x_test, y_test, z_test = get_xyz_from_dataloader(dict_load_data['testing_loader'])
    xl = dict_load_data['xl']

    x_train_KCunif_torch, y_train_KCunif_torch, z_train_KCunif_torch, x_train_KCmassage_torch, y_train_KCmassage_torch \
        = Kamiran_wrapper(x_train, y_train, z_train, xl)

    print("Begin KC massaging")
    ratio = x_train_KCmassage_torch.shape[0] // x_train.shape[0]
    train_loader_massaging = get_data_loader_from_given_data(x=x_train_KCmassage_torch,
                                                             y=y_train_KCmassage_torch,
                                                             z=z_train,
                                                             desired_batch_size=ratio * desired_batch_size)
    net = train_base_nn(train_loader=train_loader_massaging, **nn_parameters)
    net.eval()
    with torch.no_grad():
        outputs, _ = net(h=x_test, z=z_train)
        probs = torch.sigmoid(outputs)
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        c_loss_KCmassage = criterion(outputs, y_test)
        fairness_KCmassage = fairness_measures(probs,
                                               y_test,
                                               z_test,
                                               propensity_test)
    print("Finished KC massaging")


    # print("Begin KC uniform")
    # ratio = x_train_KCunif_torch.shape[0]//x_train.shape[0]
    # train_loader_massaging = get_data_loader_from_given_data(x=x_train_KCunif_torch,
    #                                                                               y=y_train_KCunif_torch,
    #                                                                               z=z_train_KCunif_torch,
    #                                                                               desired_batch_size=ratio*desired_batch_size)
    # net = train_base_nn(train_loader=train_loader_massaging, **nn_parameters)
    # net.eval()
    # outputs, _ = net(h=x_test, z=z_train)
    # probs = torch.sigmoid(outputs)
    # criterion = nn.BCELoss(reduction='mean')
    # c_loss_KCunif = criterion(outputs, y_test)
    # fairness_KCunif = fairness_measures(probs[:, 0].detach().cpu().numpy(),
    #                                     y_test[:, 0, :].detach().cpu().numpy(),
    #                                     z_test[:, 0].detach().cpu().numpy(),
    #                                     propensity_test)
    # print("Finished KC uniform")

    return c_loss_KCmassage, fairness_KCmassage

def train_base_nn(train_loader,
                  nr_of_layers,
                  nr_io,
                  dropout_probability,
                  desired_number_of_epochs=500,
                  learning_rate=0.001,
                  scheduler_factor=None,
                  scheduler_patience=None):


    initial_network = debiasing_networks.Net(train_loader=train_loader,
                                             nr_of_layers=nr_of_layers,
                                             nr_io=nr_io,
                                             dropout_probability=dropout_probability)
    net = copy.deepcopy(initial_network)

    # create an optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # create a step-size scheduler
    scheduler = custom_lr_scheduler.CustomReduceLROnPlateau(optimizer, 'min',
                                                            verbose=True,
                                                            factor=scheduler_factor,
                                                            patience=scheduler_patience,
                                                            eps=1e-6)
    criterion = nn.BCEWithLogitsLoss(reduction='sum')

    nr_training = train_loader.dataset.x.size()[2]

    for epoch in range(desired_number_of_epochs):

        net.train()

        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):

            # get the inputs
            x_sample, y_sample, z_sample, _ = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, _ = net(h=x_sample)
            c_loss = criterion(outputs, y_sample.view_as(outputs))
            running_loss += c_loss/nr_training

            c_loss.backward()
            optimizer.step()

        # between epochs
        scheduler.step(running_loss.item())

        if scheduler.has_convergence_been_reached():
            print('INFO: Converence has been reached. Stopping iterations.')
            break

    return net

# Comparison of proposed deep debiasing method with other existing methods
def main(taskid):

    # unravel dataset_num and mc from taskid
    # np.unravel_index(taskid, [a, b])
    #   a = number of dataset_num's
    #   b = number of Monte Carlo realisations of training/testing splits
    taskid = int(taskid[0])
    dataset_num, mc = np.unravel_index(taskid, [3, 100])
    np.random.seed(1011)
    desired_seeds = np.random.randint(0, 2 ** 32, size=(100, 3), dtype=np.uint32)
    print('dataset number {}'.format(dataset_num))
    print('monte carlo sim number {}'.format(mc))
    desired_seed = desired_seeds[mc, dataset_num]

    # anytime a neural network is trained, the following parameters are used throughout
    nn_parameters_base = {'scheduler_factor': 0.9,
                     'scheduler_patience': 10,
                     'learning_rate': 0.001,
                     'desired_number_of_epochs': 500,
                     'dropout_probability': 0.2}

    nn_parameters = copy.deepcopy(nn_parameters_base)

    if dataset_num == 0:  # UCI adult dataset has both gender and race protected attributes

        dataset_id = 'F_A_gender'
        testing_percent = 50
        desired_batch_size = 1000
        proj_loss_weights = torch.cat((torch.Tensor([0.0]), torch.log(torch.linspace(np.exp(0.05), np.exp(0.5), 10)), torch.log(torch.linspace(np.exp(0.6), np.exp(0.95), 3)), torch.Tensor([1.0])))
        nn_parameters_base.update({
                              'nr_io': 32,
                              'nr_of_layers': 10,
                              })

        nn_parameters.update({'debias_type':'causal_inference',
                              'nr_io': 32,
                              'nr_of_layers': 10,
                              'debias_individual_layers_index': None,
                              'scalarization':'chebychev'})

    if dataset_num == 1:  # UCI adult dataset has both gender and race protected attributes

        dataset_id = 'F_A_race'
        testing_percent = 50
        desired_batch_size = 1000
        proj_loss_weights = torch.cat((torch.Tensor([0.0]), torch.log(torch.linspace(np.exp(0.05), np.exp(0.3), 10)), torch.log(torch.linspace(np.exp(0.3), np.exp(0.95), 3)), torch.Tensor([1.0])))

        nn_parameters_base.update({
                              'nr_io': 32,
                              'nr_of_layers': 10,
                              })

        nn_parameters.update({'debias_type':'causal_inference',
                              'nr_io': 32,
                              'nr_of_layers': 10,
                              'debias_individual_layers_index': None,
                              'scalarization':'chebychev'})

    elif dataset_num == 2:  # ProPublica COMPAS dataset

        dataset_id = 'F_P1'
        testing_percent = 50
        desired_batch_size = 150
        proj_loss_weights = torch.cat((torch.Tensor([0.0]), torch.log(torch.linspace(np.exp(0.05), np.exp(0.35), 10)), torch.log(torch.linspace(np.exp(0.4), np.exp(0.95), 3)), torch.Tensor([1.0])))

        nn_parameters_base.update({
                              'nr_io': 4,
                              'nr_of_layers': 4,
                              })

        nn_parameters.update({'nr_io': 4,
                              'nr_of_layers': 4,
                              'debias_individual_layers_index': None,
                              'scalarization': 'chebychev'})

    dict_load_data = debiasing.load_data_for_debiasing_sweep(
        dataset_id=dataset_id,
        number_of_continuous_y_intervals=None,
        show_data=False,
        debias_type=None,
        testing_percent=testing_percent,
        desired_seed=desired_seed,
        desired_batch_size=desired_batch_size,
        desired_number_of_samples=None)



    debiasing_obj = debiasing.Debiasing(**nn_parameters, **dict_load_data)


    # c_loss_KCmassage, fairness_KCmassage \
    #     = get_kamiran_results(dict_load_data,
    #                           desired_batch_size,
    #                           nn_parameters_base,
    #                           debiasing_obj.propensity_test)

    recorded_sweep_results_train, recorded_sweep_results_test, phats_list \
        = debiasing.debiasing_sweep(debiasing_obj,
                                    desired_seed=desired_seed,
                                    proj_loss_weights=proj_loss_weights)

    # adversarial
    lambdas = torch.linspace(0, 1, 15)
    adv_sweep_train, adv_sweep_test \
        = adversarial_sweep(dict_load_data['train_loader'],
                            dict_load_data['testing_loader'],
                            lambdas,
                            debiasing_obj.propensity_train,
                            debiasing_obj.propensity_test,
                            nn_parameters)



    # clean things up and put them in the right directory
    # os.system('mkdir compareMethods_data{}_mc{}'.format(dataset_num, mc))
    # os.system('mv -f *.pdf compareMethods_data{}_mc{}/'.format(dataset_num, mc))
    # os.system('mv figs_exp* compareMethods_data{}_mc{}/'.format(dataset_num, mc))
    # os.system('mv all*.gif compareMethods_data{}_mc{}/'.format(dataset_num, mc))

    simresults = {
        "proj_loss_weights": proj_loss_weights,
        "recorded_sweep_results_train": recorded_sweep_results_train,
        "recorded_sweep_results_test": recorded_sweep_results_test,
        "lambdas": lambdas,
        "adv_recorded_sweep_results_train": adv_sweep_train,
        "adv_recorded_sweep_results_test": adv_sweep_test,
        # "c_loss_KCmassage": c_loss_KCmassage,
        # "fairness_KCmassage": fairness_KCmassage,
        "dataset_id": dataset_id,
        "nn_parameters": nn_parameters,
        "dict_load_data": dict_load_data,
        "desired_seed": desired_seed
    }

    with open("compareMethods_data{}_mc{}.p".format(dataset_num, mc), "wb") as fp:  # Pickling
        pickle.dump(simresults, fp)


# Run the actual program
if __name__ == "__main__":
  main(sys.argv[1:])

