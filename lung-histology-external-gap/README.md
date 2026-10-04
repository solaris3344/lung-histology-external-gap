# Scanner harmonization and the external-validation gap in CT lung-cancer histology

Reproducibility repository for the study:

**Scanner harmonization did not close the external-validation gap in CT-based lung cancer histology classification: evidence of a tumour-size spectrum effect**

The study examines adenocarcinoma (ADC) versus squamous cell carcinoma (SCC) classification across two public CT cohorts:

- **Lung-PET-CT-Dx** — development cohort
- **NSCLC-Radiomics (Lung1)** — external cohort

The central question is whether an internal-to-external performance drop is mainly explained by measurable acquisition differences or by a change in the clinical/anatomical spectrum of the cohorts.

## Main analyses reproduced here

The repository contains reusable implementations for:

- external-gap decomposition into an empirical within-target reference term and a transfer term;
- Nuisance-CVA (hard and soft variants);
- simplified location-scale ComBat used in the study;
- tumour-size analyses;
- patient-level bootstrap utilities;
- slice-level versus patient-level split-policy auditing;
- ImageNet ResNet-50 feature extraction utilities;
- lesion-centred CT preprocessing helpers.

The repository does **not** redistribute TCIA imaging data or pretrained third-party model weights.

## Key reported results

Using frozen ImageNet ResNet-50 features:

- Lung-PET-CT-Dx internal AUC: **0.843**
- Dx → NSCLC-Radiomics external AUC: **0.644**
- within-NSCLC-Radiomics AUC: **0.559**
- hard Nuisance-CVA external change: **ΔAUC −0.001**  
  95% CI **−0.024 to 0.023**
- acquisition-group predictability: **0.53 → 0.16**
- tumour diameter alone: AUC **0.755** internally and about **0.535** after transfer

Using the frozen CT foundation representation:

- internal AUC: **0.760**
- external AUC: **0.658**
- within-NSCLC-Radiomics AUC: **0.528**

In the split-policy audit, random slice-level splitting with whole axial slices and frozen ImageNet features produced four-class accuracy **0.998**, whereas patient separation reduced balanced accuracy to **0.273**.

These values are reference targets, not unit-test tolerances.

## Repository layout

```text
lung-histology-external-gap/
├─ README.md
├─ README_TR.md
├─ CITATION.cff
├─ requirements.txt
├─ pyproject.toml
├─ config.example.yaml
├─ data/
│  └─ README.md
├─ docs/
│  ├─ DATASETS.md
│  ├─ REPRODUCIBILITY.md
│  └─ MANUSCRIPT_MAPPING.md
├─ scripts/
│  ├─ demo_gap_decomposition.py
│  ├─ demo_nuisance_cva.py
│  └─ run_smoke_test.py
└─ src/lung_external_gap/
   ├─ __init__.py
   ├─ gap.py
   ├─ nuisance_cva.py
   ├─ combat.py
   ├─ bootstrap.py
   ├─ size_analysis.py
   ├─ split_audit.py
   ├─ ct_preprocess.py
   └─ resnet50_features.py
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Quick smoke test

```bash
python scripts/run_smoke_test.py
```

## Data

Both cohorts are available through **The Cancer Imaging Archive (TCIA)**. See `docs/DATASETS.md`.

The data are intentionally not included in this repository. Users must obtain the datasets under the original provider terms and configure local paths.

## Reproducibility notes

The manuscript used two closely related logistic-regression protocols:

- **Protocol A** — harmonization experiments: centred PCA64 + logistic regression, `C=1`.
- **Protocol B** — tumour-size analyses: standardised PCA64 + logistic regression, `C=0.1`.

Internal performance was estimated with stratified 5-fold cross-validation repeated 5 times. External confidence intervals and paired AUC differences used **2,000 patient-level bootstrap resamples**.

See `docs/REPRODUCIBILITY.md` for details.

## Third-party models

The study used:

- torchvision ImageNet ResNet-50 (`IMAGENET1K_V2`) without fine-tuning;
- a published 3D CT foundation model (FMCIB) using the authors' released weights.

Third-party weights are not redistributed here.

## Citation

See `CITATION.cff`.

## License

No software license has been selected in this public package. Add a license before encouraging downstream reuse.
