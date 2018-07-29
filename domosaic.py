#!/usr/bin/env python

import sys, h5py
import numpy as np
import scipy.ndimage as ndimage
import matplotlib
import spimage
import argparse


def deldataset(file, dataset):
    try:
        del file[dataset]
    except:
        pass

parser = argparse.ArgumentParser(prog='domosaic.py', description='Create a mosaic of all results and compute R factors.')
parser.add_argument('-c', '--coacsfile',  metavar='COACSFILE', type=str, default='invicosa72orig.mat', help='COACS results file')
parser.add_argument('phasingfiles',  nargs='*', metavar='PHASINGFILES', type=str, default=['vs72/phasing.h5', 'ossvs72/phasing.h5', 'rs72f2/phasing.h5', 'ossrs72f2/phasing.h5', 'rs72f2sh/phasing.h5', 'vs72nw/phasing.h5', 'rs72/phasing.h5'], help='Phasing results')
parser.add_argument('--f2',  nargs='*', metavar='F2', type=str, default=[0, 0, 0, 0, 0, 1, 1], help='Multiply by F2 when doing R factor')
parser.add_argument('-r', '--ref',  metavar='REF', type=str, default='reference.mat', help='File containing reference pattern')
args = parser.parse_args()


with h5py.File(args.coacsfile, 'r') as f:
    intensities = f['r3b'][:]
    f2 = np.reshape(f['f2'][:],(256,256))
    strucfactors = np.sqrt(np.fft.fftshift(intensities * f2))
    vs = np.sqrt(np.clip(f['vs'], 0, 1000))

with h5py.File(args.ref, 'r') as reff:
    f2 = np.reshape(reff['f2'][:],(256,256))
    reference = reff['reference'][:]
    reference = np.fft.ifftshift(np.fft.fft2(np.fft.fft2(np.fft.fftshift(reference['real'] + 1j * reference['imag'])) * np.sqrt(f2))) / 65536 / 65536 * 4

fs = []
images = []
for name in args.phasingfiles:
    print("Opening %s" % name)
    fs.append(h5py.File(name, 'r+'))
    images.append(fs[-1]['super_images'])

M = 50
N = len(fs)
# TODO DIMS
mosaic = np.ones((32 * M - 1, 31 * (N + 1)), dtype=np.complex128) * 0.035
Rs = np.zeros((N,M))
Rsvs = np.zeros((M,))
MSEs = np.zeros((N,M))

r_sum = []
r_min = []
r_max = []

refsubimage = reference[128-15:128+16,128-15:128+16]

for n in range(N):    
    for m in range(M):
        if n == 0:
            x = 0
            y = m
            mosaic[y*32:(y+1)*32-1,x*31:x*31+31] = refsubimage.T
        x = n + 1
        y = m
        subimage = images[n][m,128-15:128+16,128-15:128+16]
        mosaic[y*32:(y+1)*32-1,x*31:x*31+31] = subimage.T
        pattern = np.fft.ifftshift(abs(np.fft.fft2(images[n][m]) * (np.sqrt(f2) if args.f2[n] == 1 else 1)))
        centers, nom_radial = spimage.radialMeanImage(abs(pattern-strucfactors), output_r=True)
        centers, denom_radial = spimage.radialMeanImage(strucfactors, output_r=True)
        r_radial = nom_radial / (denom_radial + 1e-9)

        MSEs[n, m] = np.linalg.norm(abs(subimage) - abs(refsubimage)) / np.linalg.norm(abs(refsubimage))
        Rs[n, m] = np.sum(abs(pattern-strucfactors)) / np.sum(strucfactors);
        if n == 0:
            vspattern = np.fft.fftshift(vs[m * 256:(m + 1) * 256, 0:256] * (np.sqrt(f2) if args.f2[n] == 1 else 1))
            centers, nom_radial_vs = spimage.radialMeanImage(abs(vspattern-strucfactors), output_r=True)
            r_vsradial = nom_radial_vs / (denom_radial + 1e-9)
            Rsvs[m] = np.sum(abs(vspattern-strucfactors)) / np.sum(strucfactors);
            if m > 0:
                r_sumvs = r_sumvs + r_vsradial
            else:
                r_sumvs = r_vsradial
        if m > 0:
            r_sum[n][:] += r_radial
            r_min[n] = np.minimum(r_radial, r_min[n])
            r_max[n] = np.maximum(r_radial, r_max[n])
        else:
            r_sum.append(r_radial.copy())
            r_min.append(r_radial.copy())
            r_max.append(r_radial.copy())

# Remove existing results before adding new ones
deldataset(fs[0], 'mosaic')
fs[0]['mosaic'] = mosaic
i = 0
for f in fs:
    deldataset(f, 'r_sum')
    deldataset(f, 'r_min')
    deldataset(f, 'r_min')
    deldataset(f, 'r_max')
    deldataset(f, 'centers')
    deldataset(f, 'R')
    deldataset(f, 'MSE')
    deldataset(f, 'Rvs')
    deldataset(f, 'r_sumvs')
    f['r_sum'] = r_sum[i]
    f['r_sumvs'] = r_sumvs
    f['r_min'] = r_min[i]
    f['r_max'] = r_max[i]
    f['centers'] = centers
    f['R'] = Rs[i,:]
    f['Rvs'] = Rsvs
    f['MSE'] = MSEs[i,:]
    i += 1


