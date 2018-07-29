import h5py
import sys

def deldataset(file, dataset):
    try:
        del file[dataset]
    except:
        pass


with h5py.File(sys.argv[1], 'r+') as f:
    deldataset(f, 'real_space_final')
    deldataset(f, 'support_final')
    deldataset(f, 'fourier_space_final')

