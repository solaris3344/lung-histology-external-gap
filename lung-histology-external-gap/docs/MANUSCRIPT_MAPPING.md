# Manuscript-to-code mapping

| Manuscript component | Repository module |
|---|---|
| External-gap decomposition, Eq. (1) | `lung_external_gap.gap` |
| ComBat, Eqs. (2)–(4) | `lung_external_gap.combat` |
| Nuisance-CVA, Eqs. (5)–(8) | `lung_external_gap.nuisance_cva` |
| Class-information retention rho_B, Eq. (9) | `lung_external_gap.nuisance_cva.class_retention_ratio` |
| Patient-level bootstrap | `lung_external_gap.bootstrap` |
| Tumour-size analyses, Eq. (10) | `lung_external_gap.size_analysis` |
| Slice/patient split audit | `lung_external_gap.split_audit` |
| CT preprocessing helpers | `lung_external_gap.ct_preprocess` |
| Frozen ImageNet feature utility | `lung_external_gap.resnet50_features` |

This repository is organized around the final manuscript definitions. Dataset download, DICOM series selection and institution-specific filesystem layouts are intentionally kept outside the core package.
