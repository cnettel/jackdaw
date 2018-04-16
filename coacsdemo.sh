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
matlab -nojvm -nodesktop -nosplash -r "createsupers"

python strip.py vs72/phasing.h5
h5repack -f GZIP=9 vs72/phasing.h5 vs72short.h5
rm vs72/phasing.h5

python strip.py rs72f2/phasing.h5
h5repack -f GZIP=9 rs72f2/phasing.h5 rs72f2short.h5
rm rs72f2/phasing.h5
