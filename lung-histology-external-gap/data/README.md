# Data directory

No patient data are distributed with this repository.

Expected cohorts:

1. Lung-PET-CT-Dx
2. NSCLC-Radiomics (Lung1)

Obtain both datasets from The Cancer Imaging Archive (TCIA) under their original terms.

The repository intentionally excludes:

- DICOM files;
- RTSTRUCT files;
- derived patient-level features;
- pretrained third-party model weights.

For a local analysis, store data outside the Git repository and configure paths in a private copy of `config.example.yaml`.
