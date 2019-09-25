
# import all our own modules
import debiasing_networks
import visualize_data
import dataset_factory
import matplotlib.pyplot as plt

desired_number_of_samples = 5000
debias_type = 'regression_cond_y' #'lda' #'regression_cond_x' #''lda' # 'regression_cond_y'
use_continuous_y = False
# number of y intervals if a continous variable is used as output
number_of_continuous_y_intervals = None

dataset_nr = 5
sim_dataset_names = ['SJL', 'ZafarWWW2017CaseII', 'ZafarAISTATS2017', 'JLcontinuousY','GMcontinuousY','gaussian_test']
sim_dataset_name = sim_dataset_names[dataset_nr]

current_dataset, x, y, z, xl, yl, zl = dataset_factory.get_dataset(simulator_dataset_name=sim_dataset_name,nr_of_samples=desired_number_of_samples,
                                                                       debias_type=debias_type,use_continuous_y=use_continuous_y,number_of_continuous_y_intervals=number_of_continuous_y_intervals)

y_val_ranges = [0.0,1.0]
group_indices_dict = dataset_factory.compute_group_indices(y=y,y_val_ranges=y_val_ranges)
group_indices = []
for k in group_indices_dict:
    group_indices.append(group_indices_dict[k])

# create a list of group indices (this is what the data loader would typically do)

#debias_layer = debiasing_networks.DebiasLayer(nr_inputs=2,nr_outputs=2,debias_type=debias_type,use_batch_normalization=False,
#                                              use_protected_projection=False,use_only_projection_penalty=True, sigma_reg=0.0)

proj_layer = debiasing_networks.ProtectedRegressionLayerCondY()

h_projected,projection_loss,w = proj_layer(h=x,z=z,y=y,turn_protected_projection_on=True, use_only_projection_penalty=False,current_epoch=None, group_indices=group_indices)

plt.clf()
xrange,yrange=visualize_data.visualize_labels(x[:,0,:].cpu().numpy(),y.cpu().numpy(),z.cpu().numpy())
# now plot the separation direction

xmid = 0.5*(xrange[0]+xrange[1])
ymid = 0.5*(yrange[0]+yrange[1])

sf = 3

x_vals = [xmid-sf*w[0],xmid+sf*w[0]]
y_vals = [ymid-sf*w[1],ymid+sf*w[1]]

plt.plot(x_vals,y_vals,'k-')
plt.xlim(xrange[0],xrange[1])
plt.ylim(yrange[0],yrange[1])
plt.axis('equal')
plt.show()

plt.clf()
visualize_data.visualize_labels(h_projected[:,0,:].cpu().numpy(),y.cpu().numpy(),z.cpu().numpy(),xrange_lim=xrange,yrange_lim=yrange)
plt.plot(x_vals,y_vals,'k-')
plt.xlim(xrange[0],xrange[1])
plt.ylim(yrange[0],yrange[1])
plt.axis('equal')
plt.show()