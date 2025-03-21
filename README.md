### DEEP HIGH-RES POLARISATION WITH LOFAR

To perform deep high-resolution polarisation studies, using multiple observations of the same field of view, we need to align observations to the same RM angle.
This requires data that has been corrected for scalar-phasediff (RR-LL) and full-Jones corrections (see Section 3.2.3 from de Jong et al. 2024; <https://arxiv.org/pdf/2407.13247>).

### Steps

The following steps can be used for polarisation alignment on a known high S/N polarised source:
1) Run `source RM_synth.sh <MS_SOURCE>` \
This step runs RM synthesis and returns a `RMsynth_ref_output` output directory with PNG inspection plots, `RMsynthFDF_*.fits` images, and `wsclean` images. This should be applied on a reference observation (select one of your observations).
2) Inspect `RMsynthFDF_maxPI.fits` output image to find a bright polarised region and draw a DS9 region file around this area.
3) Run `source get_polalign.sh -region <DS9_regionfile> -in_dir <INPUT_DIRECTORY> -out_dir <OUTPUT_DIRECTORY>` \
This step uses the reference observation (on which you ran RM synthesis) and the DS9 region file input from the previous step to obtain the reference values for the RM in CSV format.
The `<INPUT_DIRECTORY>` should be the directory with the RM synthesis output, which is by default called `RMsynth_ref_output`.
4) Run `source apply_polalign.sh -region <DS9_regionfile> -ms <INPUT_MS_ARRAY> -rm_csv <RM_CSV>` \
This step obtains and applies the polarisation alignment corrections, using the DS9 region file and the RM reference values from the previous step. Note that `-ms` can be many meusurement sets at the same, since the script will loop over these.
The final output is folder called `output`, which has the MeasurementSets with the corrections applied and the corrections in h5parm format. These h5parm corrections are scalar values constant over time but with a frequency variability.

#### Future improvements:
1) Testing code on different data.
2) Automatic polarisation region detection (automatic DS9 drawing).
3) Automated workflow that detects polarised sources and aligns different observations automatically.
