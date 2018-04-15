#!/bin/bash

matlab -nojvm -nodesktop -nosplash -r "replicates"

# Phase coacsed patterns
mkdir vs72
python phasing.py --output vs72/ --variable vs --filename invicosa72orig.mat

# Phase original ("rs") patterns
# Apply mask for the negative values, apply Hann window (in f2)
mkdir rs72f2
python phasing.py --output rs72f2/ --variable rs --f2 --mask --filename invicosa72orig.mat

python processprtfs.py --phase vs72/phasing.h5 --ref reference.mat
python processprtfs.py --phase rs72f2/phasing.h5 --ref reference.mat

# Do mosaic and compute R factors in both modes
python domosaic.py --coacsfile invicosa72orig.mat --coacsphase vs72/phasing.h5 --origphase rs72f2/phasing.h5 --dor a
python domosaic.py --coacsfile invicosa72orig.mat --coacsphase vs72/phasing.h5 --origphase rs72f2/phasing.h5 --dor b

# Create more manageable files