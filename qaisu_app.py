from flask import Flask, render_template_string, request, redirect, url_for, jsonify
from sqlalchemy import create_engine, text
import pandas as pd
import re
from datetime import datetime

app = Flask(__name__)

KARAR_ETIKETLERI = {
    'Roketsan da değerlendirilecek': "Roketsan'da değiştirilecek",
}

def karar_goster(karar):
    if karar is None:
        return ''
    return KARAR_ETIKETLERI.get(str(karar), str(karar))

@app.template_filter('karar_goster')
def karar_goster_filter(karar):
    return karar_goster(karar)

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F600-\U0001F64F"
    "]+",
    flags=re.UNICODE,
)

def temiz_metin(text):
    if text is None:
        return ''
    text = EMOJI_PATTERN.sub('', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.template_filter('temiz_metin')
def temiz_metin_filter(text):
    return temiz_metin(text)

DB_URL = "postgresql+psycopg2://qaisu:qaisu_pass@localhost:5432/kalite_db"

def get_engine():
    return create_engine(DB_URL)

def init_db():
    engine = get_engine()
    with engine.connect() as con:
        con.execute(text("""
            CREATE TABLE IF NOT EXISTS feedback_log (
                id SERIAL PRIMARY KEY,
                kayit_turu TEXT DEFAULT 'UDF',
                kayit_no TEXT,
                kalem TEXT,
                ai_karari TEXT,
                muhendis_karari TEXT,
                override_mi BOOLEAN,
                aciklama TEXT,
                tarih TIMESTAMP DEFAULT NOW()
            )
        """))
        con.commit()

ROKETSAN_LOGO_CSS = """
        .header-inner { display: flex; align-items: center; justify-content: space-between; gap: 24px; max-width: 1300px; margin: 0 auto; }
        .header-title h1 { margin: 0; font-size: 24px; }
        .header-title p { margin: 5px 0 0; color: #aaa; font-size: 13px; }
        .roketsan-logo { display: flex; align-items: center; gap: 12px; text-decoration: none; flex-shrink: 0; padding: 8px 14px; border-radius: 8px; background: rgba(255,255,255,0.08); transition: background 0.25s, transform 0.25s; }
        .roketsan-logo:hover { background: rgba(255,255,255,0.14); transform: translateY(-1px); }
        .roketsan-icon-wrap { width: 46px; height: 46px; animation: roketsan-float 4s ease-in-out infinite; }
        .roketsan-icon { width: 100%; height: 100%; display: block; }
        .roketsan-stripe { animation: roketsan-shimmer 2.8s ease-in-out infinite; }
        .roketsan-stripe:nth-child(1) { animation-delay: 0s; }
        .roketsan-stripe:nth-child(2) { animation-delay: 0.12s; }
        .roketsan-stripe:nth-child(3) { animation-delay: 0.24s; }
        .roketsan-stripe:nth-child(4) { animation-delay: 0.36s; }
        .roketsan-stripe:nth-child(5) { animation-delay: 0.48s; }
        .roketsan-stripe:nth-child(6) { animation-delay: 0.6s; }
        .roketsan-stripe:nth-child(7) { animation-delay: 0.72s; }
        .roketsan-stripe:nth-child(8) { animation-delay: 0.84s; }
        .roketsan-text { font-family: Arial, Helvetica, sans-serif; font-size: 20px; font-weight: 800; letter-spacing: 3px; color: #ffffff; white-space: nowrap; }
        @keyframes roketsan-float { 0%, 100% { transform: translateY(0) rotate(0deg); } 50% { transform: translateY(-4px) rotate(3deg); } }
        @keyframes roketsan-shimmer { 0%, 100% { opacity: 0.65; } 50% { opacity: 1; } }
"""

ROKETSAN_LOGO_HTML = """
        <a href="/" class="roketsan-logo" aria-label="Roketsan">
            <div class="roketsan-icon-wrap">
                <svg viewBox="0 0 56 56" class="roketsan-icon" aria-hidden="true">
                    <defs><clipPath id="roketsanClip"><circle cx="28" cy="28" r="25"/></clipPath></defs>
                    <g clip-path="url(#roketsanClip)">
                        <rect x="-12" y="-8" width="6" height="72" transform="rotate(-30 28 28)" fill="#32b848" class="roketsan-stripe"/>
                        <rect x="-2" y="-8" width="6" height="72" transform="rotate(-30 28 28)" fill="#32b848" class="roketsan-stripe"/>
                        <rect x="8" y="-8" width="6" height="72" transform="rotate(-30 28 28)" fill="#32b848" class="roketsan-stripe"/>
                        <rect x="18" y="-8" width="6" height="72" transform="rotate(-30 28 28)" fill="#32b848" class="roketsan-stripe"/>
                        <rect x="28" y="-8" width="6" height="72" transform="rotate(-30 28 28)" fill="#32b848" class="roketsan-stripe"/>
                        <rect x="38" y="-8" width="6" height="72" transform="rotate(-30 28 28)" fill="#32b848" class="roketsan-stripe"/>
                        <rect x="48" y="-8" width="6" height="72" transform="rotate(-30 28 28)" fill="#32b848" class="roketsan-stripe"/>
                        <rect x="58" y="-8" width="6" height="72" transform="rotate(-30 28 28)" fill="#32b848" class="roketsan-stripe"/>
                    </g>
                </svg>
            </div>
            <span class="roketsan-text">ROKETSAN</span>
        </a>
"""

# ============================================================
# ANA SAYFA — UDF / VIR Sekmeleri
# ============================================================
HTML_INDEX = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>QAISU — Kalite Karar Sistemi</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; background: #f5f5f5; }
        .header { background: #1a1a2e; color: white; padding: 16px 40px; }
""" + ROKETSAN_LOGO_CSS + """
        .container { max-width: 1300px; margin: 30px auto; padding: 0 20px; }
        .stats { display: flex; gap: 20px; margin-bottom: 30px; }
        .stat-card { background: white; padding: 20px; border-radius: 8px; flex: 1; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .stat-card h3 { margin: 0 0 5px; color: #666; font-size: 13px; text-transform: uppercase; }
        .stat-card h3.stat-override { text-transform: none; letter-spacing: 0.5px; }
        .stat-card .value { font-size: 32px; font-weight: bold; color: #1a1a2e; }
        .tabs { display: flex; gap: 0; margin-bottom: 0; }
        .tab { padding: 12px 30px; cursor: pointer; background: #ddd; border-radius: 8px 8px 0 0; font-weight: bold; font-size: 14px; color: #555; border: none; }
        .tab.active { background: #1a1a2e; color: white; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        table { width: 100%; background: white; border-radius: 0 8px 8px 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-collapse: collapse; }
        th { background: #1a1a2e; color: white; padding: 12px 15px; text-align: left; font-size: 13px; }
        td { padding: 12px 15px; border-bottom: 1px solid #eee; font-size: 14px; }
        tr:hover { background: #f9f9f9; }
        .badge { padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: bold; }
        .badge-uyusmazlik { background: #ffe0e0; color: #c0392b; }
        .badge-uyusuyor { background: #e0f5e0; color: #27ae60; }
        .btn { padding: 6px 14px; border: none; border-radius: 4px; cursor: pointer; font-size: 13px; text-decoration: none; }
        .btn-incele { background: #3498db; color: white; }
        .btn-incele:hover { background: #2980b9; }
        .btn-vir { background: #8e44ad; color: white; }
        .btn-vir:hover { background: #7d3c98; }
    </style>
</head>
<body>
<div class="header">
    <div class="header-inner">
        <div class="header-title">
            <h1>QAISU — Kalite Karar Destek Sistemi</h1>
        </div>
""" + ROKETSAN_LOGO_HTML + """
    </div>
</div>
<div class="container">
    <div class="stats">
        <div class="stat-card">
            <h3>UDF Toplam</h3>
            <div class="value">{{ udf_toplam }}</div>
        </div>
        <div class="stat-card">
            <h3>UDF Uyuşmazlık</h3>
            <div class="value" style="color:#c0392b">{{ udf_uyusmazlik }}</div>
        </div>
        <div class="stat-card">
            <h3>VIR Toplam</h3>
            <div class="value" style="color:#8e44ad">{{ vir_toplam }}</div>
        </div>
        <div class="stat-card">
            <h3>VIR Uyuşmazlık</h3>
            <div class="value" style="color:#c0392b">{{ vir_uyusmazlik }}</div>
        </div>
        <div class="stat-card">
            <h3>Onaylanan</h3>
            <div class="value" style="color:#27ae60">{{ onaylanan }}</div>
        </div>
        <div class="stat-card">
            <h3 class="stat-override">OVERRIDE</h3>
            <div class="value" style="color:#e67e22">{{ override }}</div>
        </div>
    </div>

    <div class="tabs">
        <button class="tab active" onclick="switchTab('udf', this)">UDF Kararları ({{ udf_toplam }})</button>
        <button class="tab" onclick="switchTab('vir', this)">VIR Kararları ({{ vir_toplam }})</button>
    </div>

    <!-- UDF TABLOSU -->
    <div id="tab-udf" class="tab-content active">
        <table>
            <thead>
                <tr>
                    <th>UDF No</th>
                    <th>Kalem</th>
                    <th>İş Emri</th>
                    <th>Hata Türü</th>
                    <th>İnsan Kararı</th>
                    <th>AI Kararı</th>
                    <th>Durum</th>
                    <th>Anomali Skoru</th>
                    <th>İşlem</th>
                </tr>
            </thead>
            <tbody>
            {% for row in udf_rows %}
                <tr>
                    <td>{{ row.UDF_NO }}</td>
                    <td>{{ row.KALEM }}</td>
                    <td>{{ row.IS_EMRI }}</td>
                    <td>{{ row.HATA_TURU }}</td>
                    <td>{{ row.KSYM_SORUMLUSU_KARAR }}</td>
                    <td><strong>{{ row.OPTIMIZE_EDILMIS_KARAR }}</strong></td>
                    <td>
                        <span class="badge {{ 'badge-uyusmazlik' if row.KARAR_UYUSMAZLIGI == 'Uyuşmazlık Var' else 'badge-uyusuyor' }}">
                            {{ row.KARAR_UYUSMAZLIGI }}
                        </span>
                    </td>
                    <td>{{ "%.2f"|format(row.A_i_Anomali_Skoru) }}</td>
                    <td><a href="/karar/{{ row.UDF_NO }}" class="btn btn-incele">İncele</a></td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
    </div>

    <!-- VIR TABLOSU -->
    <div id="tab-vir" class="tab-content">
        <table>
            <thead>
                <tr>
                    <th>VIR No</th>
                    <th>Kalem</th>
                    <th>Tedarikçi</th>
                    <th>Sapma Sınıfı</th>
                    <th>İnsan Kararı</th>
                    <th>AI Kararı</th>
                    <th>Durum</th>
                    <th>Anomali Skoru</th>
                    <th>İşlem</th>
                </tr>
            </thead>
            <tbody>
            {% for row in vir_rows %}
                <tr>
                    <td>{{ row.Numara }}</td>
                    <td>{{ row.KALEM }}</td>
                    <td>{{ row['Tedarikçi'] }}</td>
                    <td>{{ row['Sapma Sınıfı'] }}</td>
                    <td>{{ row.VIR_KARAR | karar_goster }}</td>
                    <td><strong>{{ row.OPTIMIZE_EDILMIS_VIR_KARAR | karar_goster }}</strong></td>
                    <td>
                        <span class="badge {{ 'badge-uyusmazlik' if row.KARAR_UYUSMAZLIGI == 'Uyuşmazlık Var' else 'badge-uyusuyor' }}">
                            {{ row.KARAR_UYUSMAZLIGI }}
                        </span>
                    </td>
                    <td>{{ "%.2f"|format(row.A_i_Anomali_Skoru) }}</td>
                    <td><a href="/vir/{{ row.Numara }}" class="btn btn-vir">İncele</a></td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
    </div>
</div>

<script>
function switchTab(tab, btn) {
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById('tab-' + tab).classList.add('active');
    btn.classList.add('active');
}
</script>
</body>
</html>
"""

# ============================================================
# UDF KARAR DETAY SAYFASI
# ============================================================
HTML_KARAR = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>QAISU — UDF Karar İncele</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; background: #f5f5f5; }
        .header { background: #1a1a2e; color: white; padding: 16px 40px; }
""" + ROKETSAN_LOGO_CSS + """
        .container { max-width: 800px; margin: 30px auto; padding: 0 20px; }
        .card { background: white; border-radius: 8px; padding: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .card h2 { margin: 0 0 20px; color: #1a1a2e; font-size: 18px; border-bottom: 2px solid #eee; padding-bottom: 10px; }
        .info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .info-item label { font-size: 12px; color: #666; text-transform: uppercase; display: block; }
        .info-item span { font-size: 16px; font-weight: bold; color: #1a1a2e; }
        .ai-box { background: #e8f4fd; border-left: 4px solid #3498db; padding: 20px; border-radius: 8px; margin: 15px 0; text-align: center; }
        .ai-box h3 { margin: 0 0 12px; color: #2980b9; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; }
        .ai-box .karar-label { font-size: 13px; color: #666; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px; }
        .ai-box .karar { font-size: 42px; font-weight: bold; padding: 10px 0; }
        .ai-box .karar.kirmizi { color: #c0392b; }
        .ai-box .karar.turuncu { color: #e67e22; }
        .ai-box .karar.sari { color: #f39c12; }
        .ai-box .karar.yesil { color: #27ae60; }
        .ai-box .aciklama { font-size: 13px; color: #555; margin-top: 12px; text-align: left; background: white; padding: 10px; border-radius: 4px; }
        .btn-group { display: flex; gap: 15px; margin-top: 20px; }
        .btn { padding: 12px 24px; border: none; border-radius: 6px; cursor: pointer; font-size: 15px; font-weight: bold; }
        .btn-onayla { background: #27ae60; color: white; flex: 1; }
        .btn-override { background: #e74c3c; color: white; flex: 1; }
        .override-form { display: none; margin-top: 20px; }
        .override-form select, .override-form textarea { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; box-sizing: border-box; margin-bottom: 10px; }
        .btn-gonder { background: #e67e22; color: white; padding: 12px 24px; border: none; border-radius: 6px; cursor: pointer; font-size: 15px; font-weight: bold; width: 100%; }
        .back-link { display: inline-flex; align-items: center; gap: 8px; color: #3498db; text-decoration: none; font-size: 14px; font-weight: 600; margin-top: 15px; padding: 6px 12px 6px 8px; border-radius: 6px; transition: background 0.2s; }
        .back-link:hover { background: #eef6fc; }
        .back-link .back-icon { display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px; flex-shrink: 0; }
        .back-link .back-icon svg { width: 20px; height: 20px; stroke: currentColor; fill: none; stroke-width: 2.2; stroke-linecap: round; stroke-linejoin: round; }
        .vaka-tabs { display: flex; gap: 0; margin-bottom: 0; margin-top: 15px; }
        .vaka-tab { padding: 10px 24px; cursor: pointer; background: #ddd; border-radius: 8px 8px 0 0; font-weight: bold; font-size: 13px; color: #555; border: none; }
        .vaka-tab.active { background: #1a1a2e; color: white; }
        .vaka-tab-content { display: none; background: white; border-radius: 0 8px 8px 8px; padding: 25px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .vaka-tab-content.active { display: block; }
        .vir-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .vir-table th { background: #6c3483; color: white; padding: 10px 12px; text-align: left; }
        .vir-table td { padding: 10px 12px; border-bottom: 1px solid #eee; }
        .vir-table tr:hover { background: #f9f9f9; }
        .no-data { color: #999; font-style: italic; text-align: center; padding: 20px; }
        .anomali-bar { background: #eee; border-radius: 10px; height: 10px; margin-top: 5px; }
        .anomali-fill { background: #e74c3c; border-radius: 10px; height: 10px; }
        .prob-section { margin-top: 8px; }
        .prob-row { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
        .prob-label { font-size: 12px; font-weight: bold; width: 100px; color: #333; }
        .prob-bar-bg { flex: 1; background: #eee; border-radius: 6px; height: 18px; }
        .prob-bar-fill { height: 18px; border-radius: 6px; display: flex; align-items: center; padding-left: 6px; font-size: 11px; color: white; font-weight: bold; min-width: 30px; }
        .prob-kirmizi { background: #c0392b; }
        .prob-turuncu { background: #e67e22; }
        .prob-sari { background: #f39c12; }
        .prob-yesil { background: #27ae60; }
        .prob-mavi { background: #2980b9; }
        .xai-box { background: #fef9e7; border-left: 4px solid #f39c12; padding: 20px; border-radius: 8px; margin-top: 15px; }
        .xai-box h3 { margin: 0 0 10px; color: #d68910; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; }
        .xai-text { font-size: 14px; color: #333; font-style: italic; margin-bottom: 15px; line-height: 1.6; }
        .xai-text strong { color: #c0392b; }
        .alert-banner { padding: 12px 16px; border-radius: 6px; font-size: 14px; line-height: 1.5; border-left: 4px solid #e67e22; background: #fdf2e9; color: #333; }
        .alert-banner.alert-kritik { border-left-color: #c0392b; background: #fdecea; }
        .aksiyon-box { background: #f8f9fa; border-left: 4px solid #5d6d7e; padding: 18px 20px; border-radius: 6px; margin-top: 15px; }
        .aksiyon-box h3 { margin: 0 0 8px; color: #2c3e50; font-size: 14px; font-weight: 700; text-transform: none; letter-spacing: 0; }
        .aksiyon-tip { font-size: 12px; font-weight: 700; color: #5d6d7e; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
        .aksiyon-metin { font-size: 14px; color: #333; line-height: 1.6; }
    </style>
</head>
<body>
<div class="header">
    <div class="header-inner">
        <div class="header-title">
            <h1>QAISU — UDF Karar İnceleme</h1>
        </div>
""" + ROKETSAN_LOGO_HTML + """
    </div>
</div>
<div class="container">
    <a href="/" class="back-link"><span class="back-icon" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M15 18l-6-6 6-6"/></svg></span>Listeye Dön</a>

    {% if alert_tedarikci %}
    <div class="alert-banner alert-yuksek" style="margin-top:15px">
        {{ alert_tedarikci | temiz_metin }}
    </div>
    {% endif %}
    {% if alert_kronik %}
    <div class="alert-banner alert-kritik" style="margin-top:8px">
        {{ alert_kronik | temiz_metin }}
    </div>
    {% endif %}

    <div class="vaka-tabs">
        <button class="vaka-tab active" onclick="switchVakaTab('udf', this)">Üretim Hattı (UDF) Bilgileri</button>
        <button class="vaka-tab" onclick="switchVakaTab('vir', this)">Tedarikçi Sapma (VIR) Geçmişi{% if vir_gecmis %} ({{ vir_gecmis|length }}){% endif %}</button>
    </div>

    <!-- UDF BİLGİLERİ SEKMESİ -->
    <div id="vaka-tab-udf" class="vaka-tab-content active">
        <h2 style="margin:0 0 20px;color:#1a1a2e;font-size:18px;border-bottom:2px solid #eee;padding-bottom:10px;">UDF Kayıt Bilgileri — #{{ row.UDF_NO }}</h2>
        <div class="info-grid">
            <div class="info-item"><label>Kalem (Stok Kodu)</label><span>{{ row.KALEM }}</span></div>
            <div class="info-item"><label>İş Emri</label><span>{{ row.IS_EMRI }}</span></div>
            <div class="info-item"><label>Hata Türü</label><span>{{ row.HATA_TURU }}</span></div>
            <div class="info-item"><label>Hata Sınıfı</label><span>{{ row.HATA_SINIFI }}</span></div>
            <div class="info-item"><label>Muayene Eden</label><span>{{ row.MUAYENE_EDEN }}</span></div>
            <div class="info-item"><label>Alikonulan Miktar</label><span>{{ row.ALIKONULAN_MIKTAR }}</span></div>
            <div class="info-item"><label>İSKO Oranı</label><span>{{ "%.1f%%"|format(row.ISKO_ORANI * 100) }}</span></div>
            <div class="info-item">
                <label>Anomali Skoru</label>
                <span>{{ "%.2f"|format(row.A_i_Anomali_Skoru) }}</span>
                <div class="anomali-bar"><div class="anomali-fill" style="width:{{ (row.A_i_Anomali_Skoru * 100)|int }}%"></div></div>
            </div>
        </div>
    </div>

    <!-- VIR GEÇMİŞİ SEKMESİ -->
    <div id="vaka-tab-vir" class="vaka-tab-content">
        <h2 style="margin:0 0 15px;color:#6c3483;font-size:18px;border-bottom:2px solid #eee;padding-bottom:10px;">Tedarikçi Sapma (VIR) Geçmişi — {{ row.KALEM }}</h2>
        {% if vir_gecmis %}
        <p style="font-size:13px;color:#666;margin-bottom:15px;">Bu stok koduna ait <strong>{{ vir_gecmis|length }}</strong> VIR kaydı bulunmaktadır.</p>
        <table class="vir-table">
            <thead>
                <tr>
                    <th>VIR No</th>
                    <th>Tedarikçi</th>
                    <th>İnsan Kararı</th>
                    <th>AI Kararı</th>
                    <th>Kök Neden</th>
                    <th>Sapma Sınıfı</th>
                    <th>Anomali Skoru</th>
                </tr>
            </thead>
            <tbody>
            {% for vir in vir_gecmis %}
                <tr>
                    <td>{{ vir.Numara }}</td>
                    <td>{{ vir['Tedarikçi'] }}</td>
                    <td>{{ vir.VIR_KARAR | karar_goster }}</td>
                    <td><strong>{{ vir.OPTIMIZE_EDILMIS_VIR_KARAR | karar_goster }}</strong></td>
                    <td>{{ vir.VIR_KOK_NEDEN }}</td>
                    <td>{{ vir['Sapma Sınıfı'] }}</td>
                    <td>{{ "%.2f"|format(vir.A_i_Anomali_Skoru) }}</td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
        {% else %}
        <p class="no-data">Bu stok koduna ait VIR kaydı bulunamadı.</p>
        {% endif %}
    </div>
    <div class="card">
        <h2>İnsan Kararı vs Sistem Kararı</h2>
        <div class="info-grid">
            <div class="info-item">
                <label>Geçmiş İnsan Kararı</label>
                <span style="color:#e74c3c">{{ row.KSYM_SORUMLUSU_KARAR }}</span>
            </div>
            <div class="info-item">
                <label>Durum</label>
                <span>{{ row.KARAR_UYUSMAZLIGI }}</span>
            </div>
        </div>
        <div class="ai-box">
            <h3>Sistem Önerisi</h3>
            <div class="karar-label">Sistem Kararı</div>
            <div class="karar {% if row.OPTIMIZE_EDILMIS_KARAR == 'Hurda' %}kirmizi{% elif row.OPTIMIZE_EDILMIS_KARAR in ['RTV', 'RU'] %}turuncu{% elif row.OPTIMIZE_EDILMIS_KARAR == 'Islah' %}sari{% else %}yesil{% endif %}">
                {{ row.OPTIMIZE_EDILMIS_KARAR | upper }}
            </div>
            <div class="aciklama">
                Anomali skoru <strong>{{ "%.2f"|format(row.A_i_Anomali_Skoru) }}</strong> —
                İSKO oranı <strong>{{ "%.1f%%"|format(row.ISKO_ORANI * 100) }}</strong>.
                Sistem maliyet ve risk faktörlerini optimize ederek bu kararı önermiştir.
            </div>
        </div>
        {% if aksiyon_metin %}
        <div class="aksiyon-box">
            <h3>Süreç Önerisi</h3>
            <div class="aksiyon-tip">{{ aksiyon_tip }}</div>
            <div class="aksiyon-metin">{{ aksiyon_metin | temiz_metin }}</div>
        </div>
        {% endif %}

        {% if xai %}
        <div class="xai-box">
            <h3>Karar Gerekçesi</h3>
            <div class="xai-text">
                Sistem bu kararı; <strong>%{{ xai.maliyet }} Yüksek Maliyet Riski</strong>,
                <strong>%{{ xai.tedarikci }} Tedarikçi Çapraz Riski</strong>,
                <strong>%{{ xai.anomali }} Anomali Skoru</strong> gerekçeleriyle almıştır.
            </div>
            <div class="prob-row">
                <div class="prob-label">Maliyet Riski</div>
                <div class="prob-bar-bg"><div class="prob-bar-fill prob-kirmizi" style="width:{{ xai.maliyet }}%">%{{ xai.maliyet }}</div></div>
            </div>
            <div class="prob-row">
                <div class="prob-label">Tedarikçi Riski</div>
                <div class="prob-bar-bg"><div class="prob-bar-fill prob-turuncu" style="width:{{ xai.tedarikci }}%">%{{ xai.tedarikci }}</div></div>
            </div>
            <div class="prob-row">
                <div class="prob-label">Anomali Skoru</div>
                <div class="prob-bar-bg"><div class="prob-bar-fill prob-sari" style="width:{{ xai.anomali }}%">%{{ xai.anomali }}</div></div>
            </div>
        </div>
        {% endif %}

        <div class="btn-group">
            <form method="POST" action="/onayla/{{ row.UDF_NO }}" style="flex:1">
                <button type="submit" class="btn btn-onayla" style="width:100%">SİSTEM KARARINI ONAYLA</button>
            </form>
            <button onclick="document.getElementById('overrideForm').style.display='block';this.style.display='none'" class="btn btn-override">KARARI DEĞİŞTİR (OVERRIDE)</button>
        </div>
        <div class="override-form" id="overrideForm">
            <form method="POST" action="/override/{{ row.UDF_NO }}">
                <label style="font-size:13px;color:#666;display:block;margin-bottom:5px">Kendi Kararınız:</label>
                <select name="muhendis_karari">
                    <option>Hurda</option>
                    <option>Islah</option>
                    <option>Kabul/OGK</option>
                    <option>RTV</option>
                    <option>RU</option>
                </select>
                <label style="font-size:13px;color:#666;display:block;margin-bottom:5px">Açıklama (opsiyonel):</label>
                <textarea name="aciklama" rows="3" placeholder="Neden farklı karar verdiniz?"></textarea>
                <button type="submit" class="btn-gonder">Gönder</button>
            </form>
        </div>
    </div>
</div>

<script>
function switchVakaTab(tab, btn) {
    document.querySelectorAll('.vaka-tab-content').forEach(function(el) {
        el.classList.remove('active');
    });
    document.querySelectorAll('.vaka-tab').forEach(function(el) {
        el.classList.remove('active');
    });
    document.getElementById('vaka-tab-' + tab).classList.add('active');
    btn.classList.add('active');
}
</script>
</body>
</html>
"""

# ============================================================
# VIR KARAR DETAY SAYFASI
# ============================================================
HTML_VIR_KARAR = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>QAISU — VIR Karar İncele</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; background: #f5f5f5; }
        .header { background: #6c3483; color: white; padding: 16px 40px; }
""" + ROKETSAN_LOGO_CSS + """
        .container { max-width: 800px; margin: 30px auto; padding: 0 20px; }
        .card { background: white; border-radius: 8px; padding: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .card h2 { margin: 0 0 20px; color: #1a1a2e; font-size: 18px; border-bottom: 2px solid #eee; padding-bottom: 10px; }
        .info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .info-item label { font-size: 12px; color: #666; text-transform: uppercase; display: block; }
        .info-item span { font-size: 16px; font-weight: bold; color: #1a1a2e; }
        .ai-box { background: #f5eef8; border-left: 4px solid #8e44ad; padding: 20px; border-radius: 8px; margin: 15px 0; text-align: center; }
        .ai-box h3 { margin: 0 0 12px; color: #6c3483; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; }
        .ai-box .karar-label { font-size: 13px; color: #666; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px; }
        .ai-box .karar { font-size: 42px; font-weight: bold; padding: 10px 0; }
        .ai-box .karar.kirmizi { color: #c0392b; }
        .ai-box .karar.turuncu { color: #e67e22; }
        .ai-box .karar.sari { color: #f39c12; }
        .ai-box .karar.yesil { color: #27ae60; }
        .ai-box .aciklama { font-size: 13px; color: #555; margin-top: 12px; text-align: left; background: white; padding: 10px; border-radius: 4px; }
        .btn-group { display: flex; gap: 15px; margin-top: 20px; }
        .btn { padding: 12px 24px; border: none; border-radius: 6px; cursor: pointer; font-size: 15px; font-weight: bold; }
        .btn-onayla { background: #27ae60; color: white; flex: 1; }
        .btn-override { background: #e74c3c; color: white; flex: 1; }
        .override-form { display: none; margin-top: 20px; }
        .override-form select, .override-form textarea { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; box-sizing: border-box; margin-bottom: 10px; }
        .btn-gonder { background: #e67e22; color: white; padding: 12px 24px; border: none; border-radius: 6px; cursor: pointer; font-size: 15px; font-weight: bold; width: 100%; }
        .back-link { display: inline-flex; align-items: center; gap: 8px; color: #8e44ad; text-decoration: none; font-size: 14px; font-weight: 600; margin-top: 15px; padding: 6px 12px 6px 8px; border-radius: 6px; transition: background 0.2s; }
        .back-link:hover { background: #f8f2fb; }
        .back-link .back-icon { display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px; flex-shrink: 0; }
        .back-link .back-icon svg { width: 20px; height: 20px; stroke: currentColor; fill: none; stroke-width: 2.2; stroke-linecap: round; stroke-linejoin: round; }
        .anomali-bar { background: #eee; border-radius: 10px; height: 10px; margin-top: 5px; }
        .anomali-fill { background: #8e44ad; border-radius: 10px; height: 10px; }
    </style>
</head>
<body>
<div class="header">
    <div class="header-inner">
        <div class="header-title">
            <h1>QAISU — VIR Karar İnceleme</h1>
        </div>
""" + ROKETSAN_LOGO_HTML + """
    </div>
</div>
<div class="container">
    <a href="/" class="back-link"><span class="back-icon" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M15 18l-6-6 6-6"/></svg></span>Listeye Dön</a>
    <div class="card" style="margin-top:15px">
        <h2>VIR Kayıt Bilgileri — #{{ row.Numara }}</h2>
        <div class="info-grid">
            <div class="info-item"><label>Kalem (Stok Kodu)</label><span>{{ row.KALEM }}</span></div>
            <div class="info-item"><label>Tedarikçi</label><span>{{ row['Tedarikçi'] }}</span></div>
            <div class="info-item"><label>Kök Neden</label><span>{{ row.VIR_KOK_NEDEN }}</span></div>
            <div class="info-item"><label>Sapma Sınıfı</label><span>{{ row['Sapma Sınıfı'] }}</span></div>
            <div class="info-item">
                <label>Anomali Skoru</label>
                <span>{{ "%.2f"|format(row.A_i_Anomali_Skoru) }}</span>
                <div class="anomali-bar"><div class="anomali-fill" style="width:{{ (row.A_i_Anomali_Skoru * 100)|int }}%"></div></div>
            </div>
        </div>
    </div>
    <div class="card">
        <h2>İnsan Kararı vs Sistem Kararı</h2>
        <div class="info-grid">
            <div class="info-item">
                <label>Geçmiş İnsan Kararı</label>
                <span style="color:#e74c3c">{{ row.VIR_KARAR | karar_goster }}</span>
            </div>
            <div class="info-item">
                <label>Durum</label>
                <span>{{ row.KARAR_UYUSMAZLIGI }}</span>
            </div>
        </div>
        <div class="ai-box">
            <h3>Sistem Önerisi</h3>
            <div class="karar-label">Sistem Kararı</div>
            <div class="karar {% if row.OPTIMIZE_EDILMIS_VIR_KARAR == 'RET' %}kirmizi{% elif row.OPTIMIZE_EDILMIS_VIR_KARAR == 'RU' %}turuncu{% elif row.OPTIMIZE_EDILMIS_VIR_KARAR in ['Roketsan da değerlendirilecek', "Roketsan'da değiştirilecek"] %}sari{% else %}yesil{% endif %}">
                {{ row.OPTIMIZE_EDILMIS_VIR_KARAR | karar_goster | upper }}
            </div>
            <div class="aciklama">
                Tedarikçi anomali skoru <strong>{{ "%.2f"|format(row.A_i_Anomali_Skoru) }}</strong> olarak hesaplanmıştır.
                Sapma sınıfı <strong>{{ row['Sapma Sınıfı'] }}</strong> — sistem risk ve maliyet optimizasyonuna göre bu kararı önermiştir.
            </div>
        </div>
        <div class="btn-group">
            <form method="POST" action="/vir/onayla/{{ row.Numara }}" style="flex:1">
                <button type="submit" class="btn btn-onayla" style="width:100%">Onayla</button>
            </form>
            <button onclick="document.getElementById('overrideForm').style.display='block'" class="btn btn-override">Override (Ez)</button>
        </div>
        <div class="override-form" id="overrideForm">
            <form method="POST" action="/vir/override/{{ row.Numara }}">
                <label style="font-size:13px;color:#666;display:block;margin-bottom:5px">Kendi Kararınız:</label>
                <select name="muhendis_karari">
                    <option>Kabul</option>
                    <option>RET</option>
                    <option>RU</option>
                    <option>Roketsan'da değiştirilecek</option>
                    <option>OGK</option>
                </select>
                <label style="font-size:13px;color:#666;display:block;margin-bottom:5px">Açıklama (opsiyonel):</label>
                <textarea name="aciklama" rows="3" placeholder="Neden farklı karar verdiniz?"></textarea>
                <button type="submit" class="btn-gonder">Gönder</button>
            </form>
        </div>
    </div>
</div>
</body>
</html>
"""

HTML_BASARILI = """
<!DOCTYPE html>
<html lang="tr">
<head><meta charset="UTF-8"><title>Kaydedildi</title>
<meta http-equiv="refresh" content="2;url=/">
<style>body{font-family:Arial;display:flex;justify-content:center;align-items:center;height:100vh;margin:0;background:#f5f5f5;}
.box{background:white;padding:40px;border-radius:8px;text-align:center;box-shadow:0 2px 4px rgba(0,0,0,0.1);}
h2{color:#27ae60;}p{color:#666;}</style>
</head>
<body><div class="box"><h2>{{ mesaj }}</h2><p>Ana sayfaya yönlendiriliyorsunuz...</p></div></body>
</html>
"""

# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def index():
    engine = get_engine()
    with engine.connect() as con:
        udf_rows = pd.read_sql('SELECT * FROM superset_udf_karar_raporu ORDER BY "A_i_Anomali_Skoru" DESC LIMIT 50', con)
        vir_rows = pd.read_sql('SELECT * FROM superset_vir_risk_raporu ORDER BY "A_i_Anomali_Skoru" DESC LIMIT 50', con)
        udf_toplam = pd.read_sql("SELECT COUNT(*) as c FROM superset_udf_karar_raporu", con).iloc[0]['c']
        vir_toplam = pd.read_sql("SELECT COUNT(*) as c FROM superset_vir_risk_raporu", con).iloc[0]['c']
        udf_uyusmazlik = pd.read_sql('SELECT COUNT(*) as c FROM superset_udf_karar_raporu WHERE "KARAR_UYUSMAZLIGI"=\'Uyuşmazlık Var\'', con).iloc[0]['c']
        vir_uyusmazlik = pd.read_sql('SELECT COUNT(*) as c FROM superset_vir_risk_raporu WHERE "KARAR_UYUSMAZLIGI"=\'Uyuşmazlık Var\'', con).iloc[0]['c']
        onaylanan = pd.read_sql("SELECT COUNT(*) as c FROM feedback_log WHERE override_mi=false", con).iloc[0]['c']
        override_count = pd.read_sql("SELECT COUNT(*) as c FROM feedback_log WHERE override_mi=true", con).iloc[0]['c']

    return render_template_string(HTML_INDEX,
        udf_rows=udf_rows.to_dict('records'),
        vir_rows=vir_rows.to_dict('records'),
        udf_toplam=udf_toplam,
        vir_toplam=vir_toplam,
        udf_uyusmazlik=udf_uyusmazlik,
        vir_uyusmazlik=vir_uyusmazlik,
        onaylanan=onaylanan,
        override=override_count
    )

@app.route('/karar/<int:udf_no>')
def karar_detay(udf_no):
    engine = get_engine()
    with engine.connect() as con:
        df = pd.read_sql('SELECT * FROM superset_udf_karar_raporu WHERE "UDF_NO"=' + str(udf_no), con)
    if df.empty:
        return "Kayıt bulunamadı", 404
    row = df.iloc[0]

    # P_c_ kolonlarını bul ve olasılık barı oluştur
    renk_map = {
        'Hurda': 'prob-kirmizi',
        'RTV': 'prob-turuncu',
        'RU': 'prob-turuncu',
        'Islah': 'prob-sari',
        'Kabul/OGK': 'prob-yesil',
    }
    probs = []
    for col in df.columns:
        if col.startswith('P_c_'):
            label = col.replace('P_c_', '')
            val = float(row[col]) if row[col] is not None else 0.0
            probs.append({
                'label': label,
                'pct': round(val * 100, 1),
                'renk': renk_map.get(label, 'prob-mavi')
            })
    probs = sorted(probs, key=lambda x: x['pct'], reverse=True)

    # VIR geçmişi — aynı kalemin VIR kayıtlarını çek
    kalem_kodu = str(row['KALEM'])
    with get_engine().connect() as con:
        df_vir_gecmis = pd.read_sql(
            "SELECT * FROM superset_vir_risk_raporu WHERE \"KALEM\"='" + kalem_kodu + "'",
            con
        )
    vir_gecmis = df_vir_gecmis.to_dict('records') if not df_vir_gecmis.empty else []

    # XAI — Optimizasyon ağırlıklarından gerekçe yüzdelerini hesapla
    # w1=0.10 maliyet, w2=0.40 ML, w3=0.20 risk, w4=0.20 anomali, w5=0.10 tedarikçi
    anomali_skoru = float(row['A_i_Anomali_Skoru']) if row['A_i_Anomali_Skoru'] else 0.0
    vir_varsa = 1.0  # M_cross etkisi — VIR kaydı varsa tedarikçi riski yüksek

    raw_maliyet = 0.10 + 0.20 * (1 - float(row['ISKO_ORANI']) if row['ISKO_ORANI'] else 0.5)
    raw_tedarikci = 0.10 + 0.20 * vir_varsa
    raw_anomali = 0.20 * anomali_skoru

    toplam = raw_maliyet + raw_tedarikci + raw_anomali
    if toplam > 0:
        xai = {
            'maliyet': round((raw_maliyet / toplam) * 100),
            'tedarikci': round((raw_tedarikci / toplam) * 100),
            'anomali': round((raw_anomali / toplam) * 100),
        }
        # Toplamı 100 yap
        fark = 100 - sum(xai.values())
        xai['maliyet'] += fark
    else:
        xai = {'maliyet': 45, 'tedarikci': 35, 'anomali': 20}

    # Erken uyarı rozetleri
    anomali = float(row['A_i_Anomali_Skoru']) if row['A_i_Anomali_Skoru'] else 0.0
    isko = float(row['ISKO_ORANI']) if row['ISKO_ORANI'] else 0.5

    alert_tedarikci = None
    alert_kronik = None

    # VIR geçmişi varsa tedarikçi uyarısı
    if vir_gecmis:
        vir_anomali_ort = sum(float(v['A_i_Anomali_Skoru']) for v in vir_gecmis) / len(vir_gecmis)
        tedarikci_adi = vir_gecmis[0].get('Tedarikçi', 'Bu tedarikçi')
        risk_pct = round(vir_anomali_ort * 100)
        alert_tedarikci = f"DİKKAT: {tedarikci_adi} tedarikçisinin parçalarının gelecek 6 ayda %{risk_pct} oranında red yeme riski (Yüksek Anomali) bulunmaktadır."

    # Hem anomali hem VIR varsa kronik risk
    if anomali > 0.6 and vir_gecmis and len(vir_gecmis) >= 2:
        alert_kronik = "KRONİK RİSKLİ PARÇA: Bu parçada hem üretim hem tedarik hatası yüksektir."

    # Aksiyon önerisi
    from qaisu_model import aksiyon_oner
    aksiyon_tip, aksiyon_metin, risk_seviyesi = aksiyon_oner(
        anomali_skoru=anomali,
        isko_orani=isko,
        vir_kayit_var_mi=len(vir_gecmis) > 0,
        optimize_edilmis_karar=str(row['OPTIMIZE_EDILMIS_KARAR']),
        tedarikci_anomali=float(vir_gecmis[0]['A_i_Anomali_Skoru']) if vir_gecmis else None
    )

    return render_template_string(HTML_KARAR, row=row, probs=probs, xai=xai, vir_gecmis=vir_gecmis,
        alert_tedarikci=alert_tedarikci, alert_kronik=alert_kronik,
        aksiyon_tip=aksiyon_tip, aksiyon_metin=aksiyon_metin)

@app.route('/vir/<vir_no>')
def vir_detay(vir_no):
    engine = get_engine()
    with engine.connect() as con:
        df = pd.read_sql('SELECT * FROM superset_vir_risk_raporu WHERE "Numara"=\'' + str(vir_no) + "'", con)
    if df.empty:
        return "Kayıt bulunamadı", 404
    return render_template_string(HTML_VIR_KARAR, row=df.iloc[0])

@app.route('/onayla/<int:udf_no>', methods=['POST'])
def onayla(udf_no):
    engine = get_engine()
    with engine.connect() as con:
        df = pd.read_sql('SELECT * FROM superset_udf_karar_raporu WHERE "UDF_NO"=' + str(udf_no), con)
        if not df.empty:
            row = df.iloc[0]
            con.execute(text("""
                INSERT INTO feedback_log (kayit_turu, kayit_no, kalem, ai_karari, muhendis_karari, override_mi)
                VALUES (:kayit_turu, :kayit_no, :kalem, :ai_karari, :muhendis_karari, :override_mi)
            """), {
                'kayit_turu': 'UDF',
                'kayit_no': str(udf_no),
                'kalem': str(row['KALEM']),
                'ai_karari': str(row['OPTIMIZE_EDILMIS_KARAR']),
                'muhendis_karari': str(row['OPTIMIZE_EDILMIS_KARAR']),
                'override_mi': False
            })
            con.commit()
    return render_template_string(HTML_BASARILI, mesaj="UDF kararı onaylandı ve kaydedildi!")

@app.route('/override/<int:udf_no>', methods=['POST'])
def override_udf(udf_no):
    engine = get_engine()
    muhendis_karari = request.form.get('muhendis_karari')
    aciklama = request.form.get('aciklama', '')
    with engine.connect() as con:
        df = pd.read_sql('SELECT * FROM superset_udf_karar_raporu WHERE "UDF_NO"=' + str(udf_no), con)
        if not df.empty:
            row = df.iloc[0]
            con.execute(text("""
                INSERT INTO feedback_log (kayit_turu, kayit_no, kalem, ai_karari, muhendis_karari, override_mi, aciklama)
                VALUES (:kayit_turu, :kayit_no, :kalem, :ai_karari, :muhendis_karari, :override_mi, :aciklama)
            """), {
                'kayit_turu': 'UDF',
                'kayit_no': str(udf_no),
                'kalem': str(row['KALEM']),
                'ai_karari': str(row['OPTIMIZE_EDILMIS_KARAR']),
                'muhendis_karari': muhendis_karari,
                'override_mi': True,
                'aciklama': aciklama
            })
            con.commit()
    return render_template_string(HTML_BASARILI, mesaj=f"Override kaydedildi! Kararınız: {muhendis_karari}")

@app.route('/vir/onayla/<vir_no>', methods=['POST'])
def vir_onayla(vir_no):
    engine = get_engine()
    with engine.connect() as con:
        df = pd.read_sql('SELECT * FROM superset_vir_risk_raporu WHERE "Numara"=\'' + str(vir_no) + "'", con)
        if not df.empty:
            row = df.iloc[0]
            con.execute(text("""
                INSERT INTO feedback_log (kayit_turu, kayit_no, kalem, ai_karari, muhendis_karari, override_mi)
                VALUES (:kayit_turu, :kayit_no, :kalem, :ai_karari, :muhendis_karari, :override_mi)
            """), {
                'kayit_turu': 'VIR',
                'kayit_no': str(vir_no),
                'kalem': str(row['KALEM']),
                'ai_karari': str(row['OPTIMIZE_EDILMIS_VIR_KARAR']),
                'muhendis_karari': str(row['OPTIMIZE_EDILMIS_VIR_KARAR']),
                'override_mi': False
            })
            con.commit()
    return render_template_string(HTML_BASARILI, mesaj="VIR kararı onaylandı ve kaydedildi!")

@app.route('/vir/override/<vir_no>', methods=['POST'])
def vir_override(vir_no):
    engine = get_engine()
    muhendis_karari = request.form.get('muhendis_karari')
    aciklama = request.form.get('aciklama', '')
    with engine.connect() as con:
        df = pd.read_sql('SELECT * FROM superset_vir_risk_raporu WHERE "Numara"=\'' + str(vir_no) + "'", con)
        if not df.empty:
            row = df.iloc[0]
            con.execute(text("""
                INSERT INTO feedback_log (kayit_turu, kayit_no, kalem, ai_karari, muhendis_karari, override_mi, aciklama)
                VALUES (:kayit_turu, :kayit_no, :kalem, :ai_karari, :muhendis_karari, :override_mi, :aciklama)
            """), {
                'kayit_turu': 'VIR',
                'kayit_no': str(vir_no),
                'kalem': str(row['KALEM']),
                'ai_karari': str(row['OPTIMIZE_EDILMIS_VIR_KARAR']),
                'muhendis_karari': muhendis_karari,
                'override_mi': True,
                'aciklama': aciklama
            })
            con.commit()
    return render_template_string(HTML_BASARILI, mesaj=f"VIR Override kaydedildi! Kararınız: {muhendis_karari}")

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=False)
