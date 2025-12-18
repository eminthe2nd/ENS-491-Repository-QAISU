# ENS 491/2 Repository: QAISU v Roketsan
### Repository Content

* **Project Management:** [Trello](https://trello.com/invite/b/6916320a0378abcf558edab6/ATTI36c683ad5f7add33eb542619ffb9641fE12FBA67/roketsan-quality-analytics-intelligence)
* **Dashboard:** Model dashboard built with Streamlit (`.py`).
* **Quality Assurance Model:** Source code created in Google Colab (`.py`).
* **Data Samples:** Example input and output using randomly generated dummy data (`.xlsx` and `.csv` respectively).
* **Glossary:** Explanations of all columns in the output file (`.xlsx`).
* **Techniques Matrix:** Tentative data mining x OR techniques matrix (`.docx`).
* **Mathematical Formulations:** Detailed formulations of the OR techniques used (`.pdf`).
* **Superset Charts:** Visual chart examples from Apache Superset (`.pdf`).

### QAISU - Kalite Karar Destek Sistemi (Model 1) (Roketsan için)
Bu proje, üretim süreçlerindeki kalite kontrol kararlarını (Kabul, RTV, Hurda vb.) optimize etmek için Makine Öğrenmesi ve Yöneylem Araştırması tekniklerini birleştirir.

## Özellikler
- **ML Modelleri:** Random Forest, XGBoost, Logistic Regression.
- **Analizler:** PCA, Kümeleme (K-Means), Anomali Tespiti (Isolation Forest).
- **Optimizasyon:** Hibrit Karar Modeli (COPQ, Kapasite ve Akış dengeli).

## Nasıl Çalıştırılır?
1. Bağımlılıkları yükleyin: pip install -r requirements.txt

2. Kodu çalıştırın: Herhangi bir notebooktan .py dosyasını açıp ana fonksiyondan INPUT_FILE ismini sizin veri setinizle uyumlu yapmanız ve yüklediğiniz datasetinde NIHAI_KARAR adlı bir sütun bulunması (Kabul, RTV, Hurda vb. satırlarla doldurulmuş) gerekmektedir.
* Girdi veri setinde olması gereken sütunlar: ('PROSES_TIPI', 'NUMERIK_SONUC', 'UST_LIMIT', 'ALT_LIMIT', 'HATA_TURU', 'Grup Bşk.', 'IS_EMRI_TAMAMLANMA_STOK_YERI', 'IS_EMRI', 'KSYM_SORUMLUSU_KARAR', 'NIHAI_KARAR', 'MM', 'NUMUNE_MIKTARI', 'MUAYENE_TIPI', 'OPERASYON_NO', 'FLOW_END_DATE', 'YARATILMA_TARIHI')

3. Çıktı: Program çalıştığında, tüm tahminleri ve QA metriklerini içeren kapsamlı bir QAISU - MODEL 1 EXAMPLE OUTPUT.csv dosyası üretir.
