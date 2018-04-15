#!/usr/bin/env python
"""
A script for phasing a single diffraction pattern and calculate PRTF, now modified
to take a set of COACS patterns.

Author:       Benedikt J. Daurer (benedikt@xray.bmc.uu.se)
              Carl Nettelblad (carl.nettelblad@it.uu.se)
Last change:  February 9, 2018
"""

# Import other modules
import numpy as np
import scipy as sp
import h5py, os, sys, time, argparse, logging
import spimage

# Parse arguments
parser = argparse.ArgumentParser(prog='phasing.py', description='A script for phasing a set of COACS patterns.')
parser.add_argument('-o', '--output',  metavar='PATH', type=str, default='./', help='Output path')
parser.add_argument('-v', '--variable',  metavar='VARIABLE', type=str, default='rs', help='Variable name')
parser.add_argument('-f', '--filename', metavar='FILENAME', type=str, default='../../invicosa72orig.mat', help='Input file')
args = parser.parse_args()

# Load the pattern and the specific Hann window used
with h5py.File(args.filename, 'r') as f:
    intensities = f[args.variable][:]
    f2 = f['f2'][:]
    mask = (intensities >= -1000).astype(np.bool)

# Create out support
support_mask = mask[0:256,0:256].copy()
support_mask[:] = 0
#support_mask[113:144,113:144] = 1
support_mask[83:174,83:174] = 1
support_mask = np.fft.fftshift(support_mask)

# Phasing parameters
niter_raar = 0
niter_hio  = 50000
niter_er   = 10000
niter_total = (niter_raar + niter_hio + niter_er * 1)
beta = 0.9

# Run phasing with 5000 individual reconstructions
R = spimage.Reconstructor()
R.set_number_of_iterations(niter_total)
R.set_number_of_outputs_images(2)
R.set_number_of_outputs_scores(2)
R.set_initial_support(support_mask=support_mask)
R.set_support_algorithm("static", number_of_iterations=niter_total)
R.append_phasing_algorithm("hio", beta_init=beta, beta_final=beta, number_of_iterations=niter_hio, constraints=['enforce_positivity', 'enforce_real'])
R.append_phasing_algorithm("er",  number_of_iterations=niter_er, constraints=['enforce_positivity', 'enforce_real'])

# N unique patterns, M reconstructions of each
N = 50
M = 100

os.system('rm %s' %(args.output + '/phasing.h5'))
for n in range(N):
    R.set_intensities(np.fft.fftshift(intensities[(n*256):((n+1)*256),:])) # * np.reshape(f2,(256,256))))
    R.set_mask(np.fft.fftshift(mask[(n*256):((n+1)*256),:]))
    output = R.reconstruct_loop(M)
    print "Done Reconstructions: %d/%d" %((n+1)*M, N*M)
    with h5py.File(args.output + '/phasing.h5', 'a') as f:
        for k,v in output.iteritems():
            if isinstance(v,dict):
                for kd,vd in v.iteritems():
                    if kd not in f.keys():
                        f.create_dataset(kd, (N*M,), dtype=vd.dtype)
                    f[kd][n*M:(n+1)*M] = vd
            else:
                if k not in f.keys():
                    f.create_dataset(k, (N*M, v.shape[1], v.shape[2]), dtype=v.dtype)
                f[k][n*M:(n+1)*M] = v
