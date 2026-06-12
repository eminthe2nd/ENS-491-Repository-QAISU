import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pulp
import os
import joblib
from sqlalchemy import create_engine
from datetime import datetime

# ============================================================
# VERİTABANI BAĞLANTISI
# ============================================================
DB_URL = "postgresql+psycopg2://qaisu:qaisu_pass@localhost:5432/kalite_db"

def get_engine():
    return create_engine(DB_URL)

# ============================================================
# VERİ YÜKLEME (Excel'den — Faz 1'de değişmeyecek)
# ============================================================
file_path = 'Random_Sentetik3.xlsx'

def load_and_preprocess_data():

    # 1. UDF VERİSİ
    udf_cols = ['UDF_NO', 'KALEM', 'IS_EMRI', 'KSYM_SORUMLUSU_KARAR', 'KSYM_KOK_NEDEN',
                'HATA_TURU', 'HATA_SINIFI', 'MUAYENE_EDEN', 'ALIKONULAN_MIKTAR', 'YARATILMA_TARIHI']
    df_udf = pd.read_excel(file_path, sheet_name='Sentetik Veri 1', usecols=udf_cols)
    df_udf['KSYM_SORUMLUSU_KARAR'] = df_udf['KSYM_SORUMLUSU_KARAR'].fillna('Belirsiz')
    df_udf['KSYM_KOK_NEDEN'] = df_udf['KSYM_KOK_NEDEN'].fillna('Bilinmiyor')

    # 2. İSKO VERİSİ
    isko_cols = ['IS_EMRI', 'STOK_KODU', 'IS_EMRI_BASLANGIC_MIKTARI',
                 'NIHAI_KABUL_MIKTARI', 'TAMAMLANMA_TARIHI']
    df_isko = pd.read_excel(file_path, sheet_name='Sentetik Veri 2', usecols=isko_cols)
    df_isko['ISKO_ORANI'] = df_isko['NIHAI_KABUL_MIKTARI'] / df_isko['IS_EMRI_BASLANGIC_MIKTARI'].replace(0, np.nan)

    # 3. VIR VERİSİ
    vir_cols = ['Numara', 'Stok Kodu', 'Nihai Karar', 'Kök Neden  Kodu', 'Tedarikçi', 'Sapma Sınıfı']
    df_vir = pd.read_excel(file_path, sheet_name='Sentetik Veri3', usecols=vir_cols)
    df_vir.rename(columns={'Stok Kodu': 'KALEM', 'Kök Neden  Kodu': 'VIR_KOK_NEDEN', 'Nihai Karar': 'VIR_KARAR'}, inplace=True)
    df_isko.rename(columns={'STOK_KODU': 'KALEM'}, inplace=True)

    return df_udf, df_isko, df_vir

# ============================================================
# EDA
# ============================================================
def perform_eda(df_udf, df_isko, df_vir):
    print(f"\nUDF Veri Boyutu: {df_udf.shape}")
    print("UDF Karar Dağılımı:")
    print(df_udf['KSYM_SORUMLUSU_KARAR'].value_counts(normalize=True).head())
    print(f"\nVIR Veri Boyutu: {df_vir.shape}")
    print("VIR Karar Dağılımı:")
    print(df_vir['VIR_KARAR'].value_counts(normalize=True).head())
    ortak_kalemler = set(df_udf['KALEM']).intersection(set(df_vir['KALEM']))
    print(f"\nOrtak Stok Kodu sayısı: {len(ortak_kalemler)}")

# ============================================================
# ÖZELLİK MÜHENDİSLİĞİ VE ANOMALİ
# ============================================================
def feature_engineering_and_anomaly(df_udf, df_isko, df_vir):

    len_udf = df_udf['KALEM'].astype(str).str.strip().str.len().max()
    len_isko = df_isko['KALEM'].astype(str).str.strip().str.len().max()
    len_vir = df_vir['KALEM'].astype(str).str.strip().str.len().max()
    dinamik_zfill = max(len_udf, len_isko, len_vir)

    df_isko['KALEM'] = df_isko['KALEM'].astype(str).str.strip().str.zfill(dinamik_zfill)
    df_udf['KALEM'] = df_udf['KALEM'].astype(str).str.strip().str.zfill(dinamik_zfill)
    df_vir['KALEM'] = df_vir['KALEM'].astype(str).str.strip().str.zfill(dinamik_zfill)

    isko_avg = df_isko.groupby('IS_EMRI')['ISKO_ORANI'].mean().reset_index()
    df_udf = df_udf.merge(isko_avg, on='IS_EMRI', how='left')
    df_udf['ISKO_ORANI'] = df_udf['ISKO_ORANI'].fillna(df_udf['ISKO_ORANI'].mean())

    df_udf['HATA_TURU_FREQ'] = df_udf.groupby('HATA_TURU')['HATA_TURU'].transform('count')
    df_udf['HATA_SINIFI_FREQ'] = df_udf.groupby('HATA_SINIFI')['HATA_SINIFI'].transform('count')

    features_udf = ['ALIKONULAN_MIKTAR', 'ISKO_ORANI', 'HATA_TURU_FREQ', 'HATA_SINIFI_FREQ']
    X_udf = df_udf[features_udf].fillna(0)

    olumsuz_kararlar = ['Hurda', 'RTV', 'Islah', 'RU']
    gercek_sorun_orani = df_udf['KSYM_SORUMLUSU_KARAR'].isin(olumsuz_kararlar).mean()
    dinamik_contamination_udf = max(0.01, min(0.20, gercek_sorun_orani))
    print(f"UDF Anomali Oranı: {dinamik_contamination_udf:.3f}")

    iso_forest_udf = IsolationForest(n_estimators=100, contamination=dinamik_contamination_udf, random_state=42)
    iso_forest_udf.fit(X_udf)
    scores_udf = iso_forest_udf.decision_function(X_udf)
    scores_inverted_udf = (scores_udf.max() - scores_udf).reshape(-1, 1)
    scaler = MinMaxScaler()
    df_udf['A_i_Anomali_Skoru'] = scaler.fit_transform(scores_inverted_udf)

    olumsuz_vir_kararlari = ['RET', 'RU']
    vir_sorun_durumu = (df_vir['Sapma Sınıfı'] == 'Majör') | (df_vir['VIR_KARAR'].isin(olumsuz_vir_kararlari))
    gercek_sorun_orani_vir = vir_sorun_durumu.mean()
    dinamik_contamination_vir = max(0.01, min(0.20, gercek_sorun_orani_vir))
    print(f"VIR Anomali Oranı: {dinamik_contamination_vir:.3f}")

    df_vir['TEDARIKCI_FREQ'] = df_vir.groupby('Tedarikçi')['Tedarikçi'].transform('count')
    df_vir['SAPMA_FREQ'] = df_vir.groupby('Sapma Sınıfı')['Sapma Sınıfı'].transform('count')
    X_vir = df_vir[['TEDARIKCI_FREQ', 'SAPMA_FREQ']].fillna(0)

    iso_forest_vir = IsolationForest(n_estimators=100, contamination=dinamik_contamination_vir, random_state=42)
    iso_forest_vir.fit(X_vir)
    scores_vir = iso_forest_vir.decision_function(X_vir)
    scores_inverted_vir = (scores_vir.max() - scores_vir).reshape(-1, 1)
    df_vir['A_i_Anomali_Skoru'] = scaler.fit_transform(scores_inverted_vir)

    return df_udf, df_vir

# ============================================================
# ML MODELLERİ — EĞİT VE KAYDET (.pkl)
# ============================================================
MODEL_DIR = '/home/ubuntu/qaisu/models'

def train_ml_models_and_get_probabilities(df_udf, df_vir):

    os.makedirs(MODEL_DIR, exist_ok=True)

    le_udf = LabelEncoder()
    df_udf['KARAR_ENCODED'] = le_udf.fit_transform(df_udf['KSYM_SORUMLUSU_KARAR'].astype(str))
    udf_classes = le_udf.classes_

    features_udf = ['ALIKONULAN_MIKTAR', 'ISKO_ORANI', 'HATA_TURU_FREQ', 'HATA_SINIFI_FREQ', 'A_i_Anomali_Skoru']
    X_udf = df_udf[features_udf].fillna(0)
    y_udf = df_udf['KARAR_ENCODED']

    rf_udf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_udf.fit(X_udf, y_udf)

    # Modeli kaydet
    joblib.dump({'model': rf_udf, 'classes': udf_classes, 'features': features_udf},
                f'{MODEL_DIR}/udf_model.pkl')
    print(f"✅ UDF modeli kaydedildi: {MODEL_DIR}/udf_model.pkl")

    udf_probs = rf_udf.predict_proba(X_udf)
    for i, class_name in enumerate(udf_classes):
        df_udf[f'P_c_{class_name}'] = udf_probs[:, i]

    le_vir = LabelEncoder()
    df_vir['VIR_KARAR'] = df_vir['VIR_KARAR'].fillna('Belirsiz').astype(str)
    df_vir['KARAR_ENCODED'] = le_vir.fit_transform(df_vir['VIR_KARAR'])
    vir_classes = le_vir.classes_

    features_vir = ['TEDARIKCI_FREQ', 'SAPMA_FREQ', 'A_i_Anomali_Skoru']
    X_vir = df_vir[features_vir].fillna(0)
    y_vir = df_vir['KARAR_ENCODED']

    rf_vir = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_vir.fit(X_vir, y_vir)

    # Modeli kaydet
    joblib.dump({'model': rf_vir, 'classes': vir_classes, 'features': features_vir},
                f'{MODEL_DIR}/vir_model.pkl')
    print(f"✅ VIR modeli kaydedildi: {MODEL_DIR}/vir_model.pkl")

    vir_probs = rf_vir.predict_proba(X_vir)
    for i, class_name in enumerate(vir_classes):
        df_vir[f'P_c_{class_name}'] = vir_probs[:, i]

    return df_udf, df_vir, udf_classes, vir_classes

# ============================================================
# OPTİMİZASYON MODELİ (orijinal koddan değişmedi)
# ============================================================
def run_optimization_model(df_udf, df_vir, udf_classes, vir_classes, df_isko):
    df_opt = df_udf.copy()
    df_vir_opt = df_vir.copy()

    C_c = {'RU': 0.6, 'Hurda': 1.0, 'Kabul/OGK': 0.2, 'Islah': 0.5, 'RTV': 0.4}
    C_c = {c: C_c.get(c, 0.5) for c in udf_classes}

    vir_kalem_set = set(df_vir_opt['KALEM'])
    df_opt['VIR_KAYDI_VAR_MI'] = df_opt['KALEM'].apply(lambda x: 1 if x in vir_kalem_set else 0)

    vir_olan_risk = df_opt[df_opt['VIR_KAYDI_VAR_MI'] == 1]['KSYM_SORUMLUSU_KARAR'].isin(['Hurda', 'RET']).mean()
    vir_olmayan_risk = df_opt[df_opt['VIR_KAYDI_VAR_MI'] == 0]['KSYM_SORUMLUSU_KARAR'].isin(['Hurda', 'RET']).mean()

    if pd.isna(vir_olmayan_risk) or vir_olmayan_risk == 0:
        dinamik_m_cross = 1.1
    else:
        dinamik_m_cross = vir_olan_risk / vir_olmayan_risk

    dinamik_m_cross = max(1.0, min(3.0, dinamik_m_cross))
    print(f"\nM_cross Çarpanı: {dinamik_m_cross:.2f}x")
    df_opt['M_cross'] = df_opt['KALEM'].apply(lambda x: dinamik_m_cross if x in vir_kalem_set else 1.0)

    udf_riskli = ['Hurda', 'RTV', 'RU', 'Islah']
    udf_kok_riskleri = df_opt.groupby('KSYM_KOK_NEDEN')['KSYM_SORUMLUSU_KARAR'].apply(lambda x: x.isin(udf_riskli).mean()).to_dict()
    df_opt['R_i'] = df_opt['KSYM_KOK_NEDEN'].map(udf_kok_riskleri).fillna(0.3).clip(lower=0.1)

    vir_cost_mapping = {'Kabul': 0.1, 'RET': 1.0, 'RU': 0.6, 'Roketsan da değerlendirilecek': 0.3}
    C_c_vir = {c: vir_cost_mapping.get(c, 0.5) for c in vir_classes}

    vir_riskli = ['RET', 'RU']
    vir_kok_riskleri = df_vir_opt.groupby('VIR_KOK_NEDEN')['VIR_KARAR'].apply(lambda x: x.isin(vir_riskli).mean()).to_dict()
    df_vir_opt['R_j'] = df_vir_opt['VIR_KOK_NEDEN'].map(vir_kok_riskleri).fillna(0.3).clip(lower=0.1)

    weights = load_dynamic_weights()
    w1, w2, w3, w4, w5 = weights['w1'], weights['w2'], weights['w3'], weights['w4'], weights['w5']
    print(f"⚙️ Kullanılan ağırlıklar: w1={w1} w2={w2} w3={w3} w4={w4} w5={w5}")

    prob = pulp.LpProblem("Birlesik_Kalite_Karar_Optimizasyonu", pulp.LpMinimize)

    x_udf = pulp.LpVariable.dicts("x_udf", ((i, c) for i in df_opt.index for c in udf_classes), cat='Binary')
    x_vir = pulp.LpVariable.dicts("x_vir", ((j, c) for j in df_vir_opt.index for c in vir_classes), cat='Binary')
    y = pulp.LpVariable.dicts("y_j", (j for j in df_vir_opt.index), lowBound=0, upBound=1, cat='Continuous')

    for i in df_opt.index:
        prob += pulp.lpSum([x_udf[(i, c)] for c in udf_classes]) == 1, f"Tek_Karar_UDF_{i}"
    for j in df_vir_opt.index:
        prob += pulp.lpSum([x_vir[(j, c)] for c in vir_classes]) == 1, f"Tek_Karar_VIR_{j}"

    ilgili_is_emirleri = df_opt['IS_EMRI'].unique()
    toplam_baslangic = df_isko[df_isko['IS_EMRI'].isin(ilgili_is_emirleri)]['IS_EMRI_BASLANGIC_MIKTARI'].sum()
    if toplam_baslangic == 0:
        toplam_baslangic = df_opt['ALIKONULAN_MIKTAR'].sum() * 10

    max_fire = toplam_baslangic * 0.10
    fire_kararlari = [c for c in udf_classes if c in ['Hurda', 'RTV', 'RU']]
    df_opt['ALIKONULAN_MIKTAR'] = pd.to_numeric(df_opt['ALIKONULAN_MIKTAR'], errors='coerce').fillna(0)
    prob += pulp.lpSum([df_opt.loc[i, 'ALIKONULAN_MIKTAR'] * x_udf[(i, c)] for i in df_opt.index for c in fire_kararlari]) <= max_fire, "ISKO_Kisit"

    gecmis_is_yuku = df_opt['MUAYENE_EDEN'].value_counts().to_dict()
    dinamik_kapasite_sozlugu = {p: max(5, int(a * 1.2) + 5) for p, a in gecmis_is_yuku.items()}
    for personel in df_opt['MUAYENE_EDEN'].unique():
        personel_kayitlari = df_opt[df_opt['MUAYENE_EDEN'] == personel].index
        kapasite = dinamik_kapasite_sozlugu.get(personel, 10)
        prob += pulp.lpSum([x_udf[(i, c)] for i in personel_kayitlari for c in udf_classes]) <= kapasite, f"Kapasite_{personel}"

    kabul_benzeri_vir_kararlari = [c for c in vir_classes if c not in ['RET', 'RU']]
    for j in df_vir_opt.index:
        dinamik_risk_carpani = df_vir_opt.loc[j, 'R_j'] * df_vir_opt.loc[j, 'A_i_Anomali_Skoru']
        prob += y[j] >= pulp.lpSum([dinamik_risk_carpani * x_vir[(j, c)] for c in kabul_benzeri_vir_kararlari]), f"Etki_Tepki_{j}"

    risk_tasiyan_kararlar = [c for c in udf_classes if c in ['Kabul/OGK', 'RU']]
    vir_risk_tasiyanlar = [c for c in vir_classes if c not in ['RET']]

    Z_maliyet = pulp.lpSum([C_c[c] * df_opt.loc[i, 'M_cross'] * x_udf[(i, c)] for i in df_opt.index for c in udf_classes]) + \
                pulp.lpSum([C_c_vir[c] * 1.0 * x_vir[(j, c)] for j in df_vir_opt.index for c in vir_classes])
    Z_ml = pulp.lpSum([(1.0 - df_opt.loc[i, f'P_c_{c}']) * x_udf[(i, c)] for i in df_opt.index for c in udf_classes]) + \
           pulp.lpSum([(1.0 - df_vir_opt.loc[j, f'P_c_{c}']) * x_vir[(j, c)] for j in df_vir_opt.index for c in vir_classes])
    Z_risk = pulp.lpSum([df_opt.loc[i, 'R_i'] * x_udf[(i, c)] for i in df_opt.index for c in risk_tasiyan_kararlar]) + \
             pulp.lpSum([df_vir_opt.loc[j, 'R_j'] * x_vir[(j, c)] for j in df_vir_opt.index for c in vir_risk_tasiyanlar])
    Z_anomali = pulp.lpSum([df_opt.loc[i, 'A_i_Anomali_Skoru'] * x_udf[(i, c)] for i in df_opt.index for c in risk_tasiyan_kararlar]) + \
                pulp.lpSum([df_vir_opt.loc[j, 'A_i_Anomali_Skoru'] * x_vir[(j, c)] for j in df_vir_opt.index for c in vir_risk_tasiyanlar])
    Z_tedarikci_etki = pulp.lpSum([y[j] for j in df_vir_opt.index])

    prob += w1 * Z_maliyet + w2 * Z_ml + w3 * Z_risk + w4 * Z_anomali + w5 * Z_tedarikci_etki, "Total_Objective"

    prob.solve()
    print(f"Optimizasyon Durumu: {pulp.LpStatus[prob.status]}")

    df_opt['OPTIMIZE_EDILMIS_KARAR'] = [
        c for i in df_opt.index for c in udf_classes if pulp.value(x_udf[(i, c)]) == 1.0
    ]
    df_vir_opt['OPTIMIZE_EDILMIS_VIR_KARAR'] = [
        c for j in df_vir_opt.index for c in vir_classes if pulp.value(x_vir[(j, c)]) == 1.0
    ]

    return df_opt, df_vir_opt, prob.status

# ============================================================
# HEALTH CHECK (orijinal koddan değişmedi)
# ============================================================
def run_health_check(df_udf, df_isko, df_vir, df_opt, prob_status):
    print("\nSİSTEM SAĞLIK KONTROLÜ (HEALTH CHECK)")
    isko_nan = df_udf['ISKO_ORANI'].isna().sum()
    print(f" -> Tanımsız İSKO Oranı: {isko_nan} " + ("✅" if isko_nan == 0 else "❌"))
    udf_anomali_nan = df_udf['A_i_Anomali_Skoru'].isna().sum()
    print(f" -> UDF Anomali NaN: {udf_anomali_nan} " + ("✅" if udf_anomali_nan == 0 else "❌"))
    vir_anomali_nan = df_vir['A_i_Anomali_Skoru'].isna().sum()
    print(f" -> VIR Anomali NaN: {vir_anomali_nan} " + ("✅" if vir_anomali_nan == 0 else "❌"))
    atanmamis = df_opt['OPTIMIZE_EDILMIS_KARAR'].isna().sum()
    print(f" -> Karar verilemeyen kayıt: {atanmamis} " + ("✅" if atanmamis == 0 else "❌"))
    print(f" -> Optimizasyon: {pulp.LpStatus[prob_status]} " + ("✅" if pulp.LpStatus[prob_status] == 'Optimal' else "❌"))

# ============================================================
# EXPORT — CSV DEĞİL, DOĞRUDAN VERİTABANINA YAZ
# ============================================================
def export_to_database(df_udf, df_vir):

    engine = get_engine()

    # UDF Export sütunları — P_c_ kolonlarını da ekle
    udf_base_cols = [
        'UDF_NO', 'YARATILMA_TARIHI', 'KALEM', 'IS_EMRI', 'KSYM_SORUMLUSU_KARAR',
        'OPTIMIZE_EDILMIS_KARAR', 'HATA_TURU', 'HATA_SINIFI', 'MUAYENE_EDEN',
        'ALIKONULAN_MIKTAR', 'ISKO_ORANI', 'A_i_Anomali_Skoru'
    ]
    p_c_cols = [c for c in df_udf.columns if c.startswith('P_c_')]
    udf_export_cols = udf_base_cols + p_c_cols
    df_udf_export = df_udf[udf_export_cols].copy()
    df_udf_export['KARAR_UYUSMAZLIGI'] = df_udf_export.apply(
        lambda x: 'Uyuşmazlık Var' if str(x['KSYM_SORUMLUSU_KARAR']) != str(x['OPTIMIZE_EDILMIS_KARAR']) else 'Uyuşuyor',
        axis=1
    )
    df_udf_export['CALISTIRMA_ZAMANI'] = datetime.now()

    # VIR Export sütunları
    vir_export_cols = [
        'Numara', 'KALEM', 'VIR_KARAR', 'OPTIMIZE_EDILMIS_VIR_KARAR',
        'VIR_KOK_NEDEN', 'Tedarikçi', 'Sapma Sınıfı', 'A_i_Anomali_Skoru'
    ]
    df_vir_export = df_vir[vir_export_cols].copy()
    df_vir_export['KARAR_UYUSMAZLIGI'] = df_vir_export.apply(
        lambda x: 'Uyuşmazlık Var' if str(x['VIR_KARAR']) != str(x['OPTIMIZE_EDILMIS_VIR_KARAR']) else 'Uyuşuyor',
        axis=1
    )
    df_vir_export['CALISTIRMA_ZAMANI'] = datetime.now()

    # Veritabanına yaz (replace: her çalışmada tabloyu sıfırdan yazar)
    df_udf_export.to_sql('superset_udf_karar_raporu', engine, if_exists='replace', index=False)
    print(f"✅ UDF tablosu DB'ye yazıldı: {len(df_udf_export)} kayıt → superset_udf_karar_raporu")

    df_vir_export.to_sql('superset_vir_risk_raporu', engine, if_exists='replace', index=False)
    print(f"✅ VIR tablosu DB'ye yazıldı: {len(df_vir_export)} kayıt → superset_vir_risk_raporu")

# ============================================================
# SONUÇ DEĞERLENDİRME
# ============================================================
def evaluate_results(df_opt, df_vir_opt):
    print("\nSONUÇ DEĞERLENDİRME (UDF)")
    total = len(df_opt)
    eslesen = len(df_opt[df_opt['KSYM_SORUMLUSU_KARAR'].astype(str) == df_opt['OPTIMIZE_EDILMIS_KARAR'].astype(str)])
    print(f"Toplam: {total} | Uyuşma: {eslesen} (%{(eslesen/total)*100:.1f})")

    print("\nSONUÇ DEĞERLENDİRME (VIR)")
    total_v = len(df_vir_opt)
    eslesen_v = len(df_vir_opt[df_vir_opt['VIR_KARAR'].astype(str) == df_vir_opt['OPTIMIZE_EDILMIS_VIR_KARAR'].astype(str)])
    print(f"Toplam: {total_v} | Uyuşma: {eslesen_v} (%{(eslesen_v/total_v)*100:.1f})")

# ============================================================
# AKSİYON ÖNERİSİ FONKSİYONU
# ============================================================
def aksiyon_oner(anomali_skoru, isko_orani, vir_kayit_var_mi, optimize_edilmis_karar, tedarikci_anomali=None):
    """
    Mevcut kalite verilerini analiz ederek somut süreç aksiyonu önerir.
    Çıktı: (aksiyon_tipi, aksiyon_metni, risk_seviyesi)
    """
    risk_seviyesi = "NORMAL"
    aksiyon_tipi = "İZLE"

    # Risk skorunu hesapla
    risk_skoru = 0
    if anomali_skoru > 0.7: risk_skoru += 3
    elif anomali_skoru > 0.4: risk_skoru += 2
    else: risk_skoru += 1

    if isko_orani < 0.70: risk_skoru += 3
    elif isko_orani < 0.85: risk_skoru += 2
    else: risk_skoru += 1

    if vir_kayit_var_mi: risk_skoru += 2
    if tedarikci_anomali and tedarikci_anomali > 0.6: risk_skoru += 2

    # Karar bazlı aksiyon
    if optimize_edilmis_karar == 'Hurda':
        if risk_skoru >= 7:
            risk_seviyesi = "KRİTİK"
            aksiyon_tipi = "SCAR + ÜRETİM DURDUR"
            metin = "ACİL: Hem üretim hem tedarik kaynaklı kritik hata. Tedarikçiye derhal SCAR (Düzeltici Faaliyet) açın ve o partinin üretimini durdurun."
        elif vir_kayit_var_mi:
            risk_seviyesi = "YÜKSEK"
            aksiyon_tipi = "SCAR AÇ"
            metin = "Tedarik kaynaklı risk yüksek. Kötü hammadde hatta yansıyor. Tedarikçiye SCAR (Düzeltici Faaliyet) açın ve gelen malzeme kontrolünü sıkılaştırın."
        else:
            risk_seviyesi = "ORTA"
            aksiyon_tipi = "CAPA AÇ"
            metin = "Üretim Kaynaklı: Proses parametrelerini gözden geçirin. İlgili iş istasyonu için CAPA (Düzeltici/Önleyici Faaliyet) başlatın."

    elif optimize_edilmis_karar == 'Islah':
        if vir_kayit_var_mi and anomali_skoru > 0.5:
            risk_seviyesi = "YÜKSEK"
            aksiyon_tipi = "SCAR + İZLE"
            metin = "Parça ıslah edilebilir; tedarikçi kaynaklı risk yüksek. SCAR açın, ıslah sonrası yeniden muayene yapın."
        else:
            risk_seviyesi = "ORTA"
            aksiyon_tipi = "ISLAH + İZLE"
            metin = "Islah Önerisi: Parça kurtarılabilir. Islah işlemini tamamlayın ve bir sonraki partide proses kontrolünü artırın."

    elif optimize_edilmis_karar == 'RTV':
        risk_seviyesi = "YÜKSEK"
        aksiyon_tipi = "TEDARİKÇİ İADE"
        metin = "Tedarikçi İadesi: Parçayı tedarikçiye iade edin ve 8D raporu talep edin. Alternatif tedarikçi değerlendirmesi başlatın."

    elif optimize_edilmis_karar == 'RU':
        risk_seviyesi = "ORTA"
        aksiyon_tipi = "KOŞULLU KABUL"
        metin = "Koşullu Kullanım: Parça derogasyon kapsamında kullanılabilir. Mühendislik onayı alın ve derogasyon kaydını açın."

    else:  # Kabul/OGK
        if anomali_skoru > 0.6:
            risk_seviyesi = "DÜŞÜK"
            aksiyon_tipi = "İZLE"
            metin = "Kabul — Anomali Takibi: Parça kabul edildi ancak anomali skoru yüksek. Bir sonraki 3 partide yakın takip edin."
        else:
            risk_seviyesi = "NORMAL"
            aksiyon_tipi = "KABUL"
            metin = "Normal Akış: Parça standart kalite kriterlerini karşılıyor. Rutin süreç devam eder."

    return aksiyon_tipi, metin, risk_seviyesi


# ============================================================
# DİNAMİK AĞIRLIK GÜNCELLEME (Continuous Learning)
# ============================================================
def update_dynamic_weights():
    """
    feedback_log tablosundaki override kayıtlarını okur,
    mühendis kararlarına göre optimizasyon ağırlıklarını günceller.
    Bu fonksiyon her sabah cron job tarafından çalıştırılır.
    """
    engine = get_engine()

    try:
        with engine.connect() as con:
            df_feedback = pd.read_sql("""
                SELECT * FROM feedback_log
                WHERE override_mi = true
                AND tarih >= NOW() - INTERVAL '30 days'
            """, con)
    except Exception as e:
        print(f"⚠️ feedback_log okunamadı: {e}")
        return {'w1': 0.10, 'w2': 0.40, 'w3': 0.20, 'w4': 0.20, 'w5': 0.10}

    if len(df_feedback) == 0:
        print("ℹ️ Son 30 günde override kaydı yok, ağırlıklar değişmedi.")
        return {'w1': 0.10, 'w2': 0.40, 'w3': 0.20, 'w4': 0.20, 'w5': 0.10}

    toplam = len(df_feedback)
    print(f"🔄 {toplam} override kaydı bulundu, ağırlıklar güncelleniyor...")

    # Override oranı yüksekse ML ağırlığını düşür, risk ağırlığını artır
    override_orani = toplam / max(toplam, 100)

    w2_yeni = max(0.25, 0.40 - override_orani * 0.15)  # ML ağırlığı azalır
    w3_yeni = min(0.35, 0.20 + override_orani * 0.10)  # Risk ağırlığı artar
    w4_yeni = min(0.30, 0.20 + override_orani * 0.05)  # Anomali ağırlığı artar
    w1_yeni = 0.10
    w5_yeni = round(1.0 - w1_yeni - w2_yeni - w3_yeni - w4_yeni, 2)
    w5_yeni = max(0.05, w5_yeni)

    # Normalize et
    toplam_w = w1_yeni + w2_yeni + w3_yeni + w4_yeni + w5_yeni
    weights = {
        'w1': round(w1_yeni / toplam_w, 3),
        'w2': round(w2_yeni / toplam_w, 3),
        'w3': round(w3_yeni / toplam_w, 3),
        'w4': round(w4_yeni / toplam_w, 3),
        'w5': round(w5_yeni / toplam_w, 3),
    }

    print(f"✅ Yeni ağırlıklar: Maliyet={weights['w1']} | ML={weights['w2']} | Risk={weights['w3']} | Anomali={weights['w4']} | Tedarikçi={weights['w5']}")

    # Ağırlıkları dosyaya kaydet
    import json
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(f'{MODEL_DIR}/dynamic_weights.json', 'w') as f:
        json.dump({**weights, 'guncelleme_tarihi': datetime.now().isoformat(), 'override_sayisi': toplam}, f)

    print(f"✅ Ağırlıklar kaydedildi: {MODEL_DIR}/dynamic_weights.json")
    return weights


def load_dynamic_weights():
    """Kaydedilmiş dinamik ağırlıkları yükler, yoksa varsayılanları döner."""
    import json
    weights_path = f'{MODEL_DIR}/dynamic_weights.json'
    if os.path.exists(weights_path):
        with open(weights_path, 'r') as f:
            w = json.load(f)
        print(f"✅ Dinamik ağırlıklar yüklendi: {weights_path}")
        return w
    return {'w1': 0.10, 'w2': 0.40, 'w3': 0.20, 'w4': 0.20, 'w5': 0.10}


# ============================================================
# ANA AKIŞ
# ============================================================
if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"QAISU MODEL ÇALIŞTIRILDI: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")

    # 1. Veri Yükleme
    df_udf, df_isko, df_vir = load_and_preprocess_data()
    perform_eda(df_udf, df_isko, df_vir)

    # 2. Özellik Mühendisliği ve Anomali
    df_udf, df_vir = feature_engineering_and_anomaly(df_udf, df_isko, df_vir)

    # 3. ML Modelleri (Eğit + .pkl kaydet)
    df_udf, df_vir, udf_classes, vir_classes = train_ml_models_and_get_probabilities(df_udf, df_vir)

    # 4. Optimizasyon
    df_udf_final, df_vir_final, prob_status = run_optimization_model(df_udf, df_vir, udf_classes, vir_classes, df_isko)

    # 5. Health Check
    run_health_check(df_udf, df_isko, df_vir_final, df_udf_final, prob_status)

    # 6. Sonuç Değerlendirme
    evaluate_results(df_udf_final, df_vir_final)

    # 7. Dinamik ağırlıkları güncelle (feedback_log'dan öğren)
    update_dynamic_weights()

    # 8. Veritabanına Yaz (CSV değil!)
    export_to_database(df_udf_final, df_vir_final)

    print(f"\n✅ TÜM SÜREÇ TAMAMLANDI: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
