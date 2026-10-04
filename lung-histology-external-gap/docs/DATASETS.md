# Datasets

## Lung-PET-CT-Dx

Development cohort used for model fitting and internal validation.

Manuscript analysis set:
- 301 ADC/SCC patients
- 244 adenocarcinoma
- 57 squamous cell carcinoma

Lesion centre:
- centre of the largest annotation box

Tumour diameter:
- longer side of the largest 2D annotation box

## NSCLC-Radiomics (Lung1)

External cohort.

Manuscript analysis set:
- 203 ADC/SCC patients
- 51 adenocarcinoma
- 152 squamous cell carcinoma

Lesion centre:
- centroid of the GTV contour in RTSTRUCT

Tumour diameter:
- longest axis-aligned 3D GTV extent

## Important comparability note

The two tumour-diameter definitions are not identical. Absolute cross-cohort diameter comparisons should therefore be interpreted as approximate; the manuscript primarily compares class-specific size structure within each cohort.
