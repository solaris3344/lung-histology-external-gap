# Reproducibility details

## ImageNet feature branch

For each lesion centre:

- field of view: 64 mm × 64 mm
- output size: 224 × 224 pixels
- axial, coronal and sagittal views
- each view uses 3 channels: centre plane and planes ±2 mm
- HU window: [-1000, 400]
- scaled to [0, 1]
- frozen torchvision ResNet-50, `IMAGENET1K_V2`
- 2048-D embedding per view
- final patient representation: mean of the three view embeddings

A 128-mm field-of-view sensitivity analysis was also performed.

## FMCIB branch

Published preprocessing used in the study:

- 50 × 50 × 50 mm cube
- 1-mm isotropic resampling
- trilinear interpolation
- outside-volume value: -1024 HU
- scaling: `(HU + 1024) / 3072`
- frozen published model weights
- 4096-D output

Weights are not redistributed.

## Classifier protocols

Protocol A:
- mean-centred features
- PCA64
- L2 logistic regression
- C=1
- used for harmonization experiments

Protocol B:
- standardised PCA64
- L2 logistic regression
- C=0.1
- used for size analyses

## Internal validation

- stratified 5-fold CV
- repeated 5 times

## External uncertainty

- 2,000 patient-level bootstrap resamples
- paired comparisons reuse the same sampled patients

## Harmonization groups

Acquisition group:
`manufacturer × slice-thickness bin`

Bins:
- <=1.5 mm
- 1.5–3.5 mm
- >3.5 mm

## Nuisance-CVA

Default `n_min=3` for acquisition harmonization.
The tumour-size nuisance analysis used `n_min=5`.

Soft mode:
`alpha = 0.01 * mean(lambda_k)`

Hard mode:
orthogonal removal of the estimated nuisance subspace.

## Split-policy audit

Two settings were reported:
- lesion-centred crops, <=20 annotated slices/patient
- all annotated whole axial slices

Classifiers:
- PCA64 + 1-NN
- PCA64 + class-balanced logistic regression

Policies:
- random slice-level folds
- patient-separated folds
