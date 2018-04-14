import sys, h5py
import numpy as np
import scipy.ndimage as ndimage
import matplotlib
import spimage

filename = 'vs72b/phasing.h5'
# Open file
f = h5py.File(filename, 'r+')

recons  = f['real_space_final'][:]
fourier = f['fourier_space_final'][:]
support = f['support_final'][:]
rerror  = f['real_error'][:]
ferror  = f['fourier_error'][:]

with h5py.File('../../invicosa72orig.mat', 'r') as forig:
    f2 = np.reshape(forig['f2'][:],(256,256))


with h5py.File('../../reference.mat', 'r') as reff:
    reference = reff['reference'][:]
    reference = np.fft.ifftshift(np.fft.fft2(np.fft.fft2(np.fft.fftshift(reference['real'] + 1j * reference['imag'])) * np.sqrt(f2)))

# Loop over 50 original images
M = 50
N = 100
subsetcount = 10

allsupers = np.zeros((M, 256, 256), dtype=np.complex128)

for m in range(M):
    indices = np.argsort(rerror[(m*N):((m+1)*N)]) + m * N
    indices = indices[0:subsetcount]
    print indices
    output_prtf = spimage.prtf(recons[indices], support[indices], enantio=True, translate=True, clearsupport=True, reference=reference)
    allsupers[m,:,:] = output_prtf['super_image'] / subsetcount

try:
    del f['super_images']
except:
    pass

f['super_images'] = allsupers

