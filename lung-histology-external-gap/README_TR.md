# Akciğer histolojisi dış doğrulama çalışması

Bu depo, aşağıdaki çalışmanın tekrar üretilebilirlik kodlarını düzenlemek için hazırlanmıştır:

**Scanner harmonization did not close the external-validation gap in CT-based lung cancer histology classification: evidence of a tumour-size spectrum effect**

Çalışma iki açık TCIA kohortunu karşılaştırır:

- Lung-PET-CT-Dx: geliştirme kohortu
- NSCLC-Radiomics (Lung1): dış doğrulama kohortu

Temel analizler:

- dış doğrulama farkının ayrıştırılması,
- ComBat,
- hard/soft Nuisance-CVA,
- tümör boyutu analizi,
- hasta-düzeyi bootstrap,
- slice-level ve patient-level split denetimi,
- frozen ImageNet ResNet-50 özellikleri.

Veri setleri ve üçüncü taraf model ağırlıkları depoya eklenmez.

Kurulum:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python scripts\run_smoke_test.py
```

Makaledeki ana referans sonuçlar README.md dosyasında listelenmiştir.
