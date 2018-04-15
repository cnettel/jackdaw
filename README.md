# jackdaw
Tools for handling phase retrieval and related tasks in sparse (low-signal) X-ray single particle imaging applications, specifically Convex Optimization of Autocorrelation with Constrained Support (COACS).

## What is COACS?
COACS is an attempt to pre-process (heal) diffraction patterns before performing phase retrieval, in a convex optimization focused on the intensities, rather than the amplitudes and their complex phase. The point is that this should simplify the phase retrieval problem.


## Using the COACS demo
`healernoninv.m` is the main COACS worker. For the simulations, it's driven by `replicates.m`. `coacsdemo.sh` also includes the calls to do phasing and generate aggregate statistics.

## Dependencies
The code is dependent on libspimage and TFOCS. Some contributions have been made to these and submitted to the authors. Contact me if you are unable to replicate the results using the public versions.

## About jackdaw
Western jackdaw is a crow family bird that tends to fly in tight flocks. They are distinct individuals, but the flight of the group is characterized by the movements and distribution of the full clattering.

In single particle imaging applications, we can approach the limit where the distribution of photons on a detector is essentially stochastic. The detector is good enough to discriminate the photon count clearly, but the distribution of the "pack" of photons will be a random sampling of the intensities each time. This is a challenge for methods that assume recorded intensities to be true, or at the very least, the measurement noise to be the same in every pixel. Due to the inherent Poisson distribution of photon count with the true intensity as the rate parameter, this is not the case.

