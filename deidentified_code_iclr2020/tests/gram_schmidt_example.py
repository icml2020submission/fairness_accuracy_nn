import torch
import debiasing_networks as dn

R = torch.zeros([5, 3], dtype=torch.float32)
R.normal_()

O = dn.gram_schmidt(R)

print('R: {}'.format(R))
print('O: {}'.format(O))
print('\n')

# now testing the orthonormality properties
nr_of_columns = O.size()[1]
for i in range(nr_of_columns):
    for j in range(nr_of_columns):
        res = torch.matmul(O[:,i],O[:,j])
        print('o_{} * o_{} = {}'.format(i,j,res.item()))