### DEEP HIGH-RES POLARISATION WITH LOFAR

To perform deep high-resolution polarisation studies, using multiple observations of the same field of view, we need to align observations to the same RM angle.
This requires data that has been corrected for scalar-phasediff (RR-LL) and full-Jones corrections (see [Section 3.2.3 from de Jong et al. 2024](https://arxiv.org/pdf/2407.13247)).

### Steps

The following steps can be used for linear polarisation alignment on a known high S/N linearly polarised source:
1) Run `source scripts/imaging_03.sh <MS_SOURCE>` \
This step makes with [WSClean](https://wsclean.readthedocs.io/) a small postage stamp image on your linearly polarised target source. Run this on all datasets from all observations.
2) Run `python scripts/RMsynt.py <PARAMS>`
This step runs RM synthesis with the obtained Q and U images (step 1).
3) Inspect the `RMsynthFDF_maxPI.fits` output image to find a bright polarised region and draw a DS9 region file around this area.
4) Run `python scripts/polalign.py --region <DS9_regionfile> <+PARAMS>` \
This step calculate the RM and chi0 on a reference observation (the observation on which you ran RM synthesis) on the linear polarised signal from your DS9 region file.
5) Run `python scripts/polalign.py --region <DS9_regionfile> --msin <MS> --ref_RM <REFERENCE_RM> --ref_chi0 <REFERENCE_CHI0>` \
This step should be run on the datasets from the other observations, using the reference RM and Chi0 values from the reference observations, and using the MeasurementSet from the observation that you are trying to align.
This gives you an h5parm with the desired phase corrections that can be applied on the datasets from this observation.
6) Applying the solutions on the data can for example be done with `lofar_helpers` (https://github.com/jurjen93/lofar_helpers)

Optionally:
1) After aligning the observations, use [Sidereal Visibility Averaging](https://github.com/jurjen93/sidereal_visibility_avg) to obtain a smaller dataset with visibilities from all observations combined.
2) Run `python scripts/RMsynth.py` with the `--do_rmclean` option to also perform RM cleaning and to make a spectrum with `scripts/plot_RMclean_spectrum.py`.

If you use this code, please cite:

van Weeren et al. (2026). 
*Polarisation and Faraday rotation measure imaging at metre wavelengths with sub-arcsecond resolution: a foundational calibration strategy*. 
MNRAS, TBD, TBD. [Archive link](https://arxiv.org/pdf/2606.18333)
