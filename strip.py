import h5py

with h5py.File('phasingshort.h5', 'r+') as f:
    del f['fourier_space_final']
    del f['real_space_final']
    del f['support_final']
