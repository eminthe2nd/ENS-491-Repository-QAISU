# ENS 491/2 Repository: QAISU v Roketsan

### QAISU - Kalite Karar Destek Sistemi (Model 1)
Bu model, üretim süreçlerindeki kalite kontrol kararlarını (Kabul, RTV, Hurda vb.) optimize etmek için Makine Öğrenmesi ve Yöneylem Araştırması tekniklerini birleştirir.

## Özellikler
- **ML Modelleri:** Random Forest, XGBoost, Logistic Regression.
- **Analizler:** PCA, Kümeleme (K-Means), Anomali Tespiti (Isolation Forest).
- **Optimizasyon:** Hibrit Karar Modeli (COPQ, Kapasite ve Akış dengeli).

### Repository Content

* **Project Management:** [Trello](https://trello.com/invite/b/6916320a0378abcf558edab6/ATTI36c683ad5f7add33eb542619ffb9641fE12FBA67/roketsan-quality-analytics-intelligence)
* **Dashboard:** Model dashboard built with Streamlit (2 `.py` files).
* **Quality Assurance Model:** Source code created in Google Colab (`.py`).
* **Data Samples:** Example input and output using randomly generated dummy data (`.xlsx` and `.csv` respectively).
* **Glossary:** Explanations of all columns in the output file (`.xlsx`).
* **Techniques Matrix:** Tentative data mining x OR techniques matrix (`.docx`).
* **Mathematical Formulations:** Detailed formulations of the OR techniques used (`.pdf`).
* **Superset Charts:** Visual chart examples from Apache Superset (`.pdf`).

# QAISU Workbench (Streamlit) – Setup & Run Guide

---

## 1) Gereksinimler

### Çalışma Ortamı
- **Python:** 3.10+ (öneri: 3.10 veya 3.11)
- **OS:** macOS / Windows / Linux
- **Disk/RAM:** Dataset boyutuna bağlı (30–50MB üstü excel’lerde RAM artar)
---

## 2) Versiyon Notları

- Python 3.10 / 3.11
- streamlit 1.38.0
- pandas 2.2.2
- scikit-learn 1.5.1

## 3) Kurulum (Adım Adım)

### A) Repo’yu indirin

```bash
git clone <REPO_URL>
cd <REPO_FOLDER>
```

### B) Virtual environment oluşturun (Önerilir)

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### C) Paketleri yükleyin

#### Önerilen `requirements.txt`

```txt
streamlit==1.38.0
pandas==2.2.2
numpy==2.0.1
scikit-learn==1.5.1
altair==5.4.1
matplotlib==3.9.2
openpyxl==3.1.5
pulp==2.9.0
xgboost==2.1.1
```

Kurulum:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> Eğer `xgboost` kurulumunda hata alırsanız, bu satırı kaldırıp tekrar kurabilirsiniz.

---

## 4) Uygulamayı Çalıştırma

```bash
streamlit run app.py
```

Tarayıcıdan açın:
```
http://localhost:8501
```

---

## 5) Kullanım Akışı

### Data & Exploration
- Dataset yükleme (.xlsx / .csv)
- Preview, kolon seçimi, histogram / bar chart / boxplot

### Modeling Studio
- Model seçimi (RF / XGB / Logistic)
- Train & Evaluate
- Metrikler, Confusion Matrix, Feature Importance

### Decisions & Export
- Generate Decisions
- Sonuç tablosu
- CSV export (Superset uyumlu)

---

## 6) Sık Karşılaşılan Hatalar

### “No labeled rows for training”
- `NIHAI_KARAR` kolonu tamamen boş olabilir.

### XGBoost hatası
- requirements.txt içinden kaldırılabilir.

### Excel okunamıyor
```bash
pip install openpyxl
```

---

## 7) Kullanılan Teknolojiler

- Streamlit
- Pandas / NumPy
- Scikit-learn
- PuLP (Optimizasyon)
- Altair & Matplotlib
- XGBoost
