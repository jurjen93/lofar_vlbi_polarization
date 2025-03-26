#!/bin/bash

# Input MS
MS=$1

# Stokes IV imaging
wsclean \
-no-update-model-required \
-minuv-l 1500.0 \
-size 800 800 \
-reorder \
-weight briggs -1.5 \
-parallel-reordering 4 \
-mgain 0.8 \
-data-column DATA \
-channels-out 96 \
-auto-mask 2.5 \
-auto-threshold 0.5 \
-pol iv \
-gridder wgridder \
-wgridder-accuracy 0.0001 \
-use-differential-lofar-beam \
-facet-beam-update 120 \
-name pol \
-scale 0.075arcsec \
-join-channels \
-nmiter 12 \
-niter 30000 \
${MS}

# Stokes QU imaging
wsclean \
-no-update-model-required \
-minuv-l 1500.0 \
-size 800 800 \
-reorder \
-weight briggs -1.5 \
-parallel-reordering 4 \
-mgain 0.8 \
-data-column DATA \
-channels-out 96 \
-auto-mask 2.5 \
-auto-threshold 0.5 \
-pol qu \
-gridder wgridder \
-wgridder-accuracy 0.0001 \
-use-differential-lofar-beam \
-facet-beam-update 120 \
-name pol \
-scale 0.075arcsec \
-join-polarizations \
-join-channels \
-squared-channel-joining \
-nmiter 10 \
-niter 20000 \
${MS}

# Cleanup
rm *-0???-*residual*.fits
rm *-0???-*psf*.fits
rm *-0???-*model*.fits
rm *-0???-*dirty*.fits
