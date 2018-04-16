#!/usr/bin/env python

import sys, h5py
import numpy as np
import scipy.ndimage as ndimage
import matplotlib
import spimage
import argparse

parser = argparse.ArgumentParser(prog='domosaic.py', description='Create a mosaic of all results and compute R factors.')
parser.add_argument('-c', '--coacsfile',  metavar='COACSFILE', type=str, default='invicosa72orig.mat', help='COACS results file')
parser.add_argument('-a', '--coacsphase',  metavar='COACSPHASE', type=str, default='vs72/phasing.h5', help='Phasing results for coacsed patterns')
parser.add_argument('-b', '--origphase', metavar='ORIGPHASE', type=str, default='rs72f2/phasing.h5', help='Phasing result for non-coacsed patterns')
parser.add_argument('-r', '--dor', metavar='DOR', type=str, default='a', help='Compute R factors for coacs (a) or orig (b)')
args = parser.parse_args()


with h5py.File(args.coacsfile, 'r') as f:
    intensities = f['r3b'][:]
    f2 = np.reshape(f['f2'][:],(256,256))
    strucfactors = np.sqrt(np.fft.fftshift(intensities * f2))
    vs = np.sqrt(np.clip(f['vs'], 0, 1000))

f1 = h5py.File(args.coacsphase, 'r+')
images1 = f1['super_images']

f2 = h5py.File(args.origphase, 'r+')
images2 = f2['super_images']

M = 50
mosaic = np.ones((319,314), dtype=np.complex128) * 0.035
Rs = np.zeros((M))
Rsvs = np.zeros((M))

if args.dor == 'a':
    rf = f1
    rimages = images1
else:
    rf = f2
    rimages = images2

for m in range(M):
    x = m % 5
    y = m / 5
    mosaic[y*32:(y+1)*32-1,x*63:x*63+31] = images1[m,128-15:128+16,128-15:128+16]
    mosaic[y*32:(y+1)*32-1,x*63+31:x*63+62] = images2[m,128-15:128+16,128-15:128+16]
    pattern = np.fft.ifftshift(abs(np.fft.fft2(rimages[m])))
    vspattern = np.fft.fftshift(vs[m * 256:(m + 1) * 256, 0:256])
    centers, nom_radial = spimage.radialMeanImage(abs(pattern-strucfactors), output_r=True)
    centers, nom_radial_vs = spimage.radialMeanImage(abs(vspattern-strucfactors), output_r=True)
    centers, denom_radial = spimage.radialMeanImage(strucfactors, output_r=True)
    r_radial = nom_radial / (denom_radial + 1e-9)
    r_vsradial = nom_radial_vs / (denom_radial + 1e-9)
    Rs[m] = np.sum(abs(pattern-strucfactors)) / np.sum(strucfactors);
    Rsvs[m] = np.sum(abs(vspattern-strucfactors)) / np.sum(strucfactors);
    if m > 0:
        r_sum = r_sum + r_radial
        r_sumvs = r_sumvs + r_vsradial
        r_min = np.minimum(r_radial, r_min)
        r_max = np.maximum(r_radial, r_max)
    else:
        r_sum = r_radial
        r_min = r_radial
        r_max = r_radial
        
        r_sumvs = r_vsradial

# Remove existing results before adding new ones
try:
    del f1['mosaic']
except:
    pass

try:
    del rf['r_sum']
except:
    pass

try:
    del f1['r_sumvs']
except:
    pass

try:
    del rf['r_min']
except:
    pass

try:
    del rf['r_max']
except:
    pass

try:
    del rf['centers']
except:
    pass

try:
    del rf['R']
except:
    pass

try:
    del f1['Rvs']
except:
    pass

f1['mosaic'] = mosaic
rf['r_sum'] = r_sum
f1['r_sumvs'] = r_sumvs

rf['r_min'] = r_min
rf['r_max'] = r_max
rf['centers'] = centers
rf['R'] = Rs

f1['Rvs'] = Rsvs

