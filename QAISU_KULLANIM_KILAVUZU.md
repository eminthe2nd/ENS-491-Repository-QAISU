# QAISU — Flask Uygulaması Kullanım Kılavuzu

Bu kılavuz, **QAISU Kalite Karar Destek Sistemi**'nin nasıl çalıştığını, Flask uygulamasının nasıl açılacağını ve ekranların ne anlama geldiğini adım adım açıklar.

---

## 1. Sistem Nedir?

QAISU, kalite kararlarını (UDF ve VIR kayıtları) bir web arayüzünde gösteren bir **Flask** uygulamasıdır.

| Bileşen | Dosya | Görevi |
|---------|-------|--------|
| Web arayüzü | `qaisu_app_v9.py` (sunucuda: `qaisu_app.py`) | Sayfaları gösterir, kullanıcı kararlarını kaydeder |
| Model / öneri motoru | `qaisu_model_v4.py` (sunucuda: `qaisu_model.py`) | Süreç önerisi (`aksiyon_oner`) üretir |
| Veritabanı | PostgreSQL (`kalite_db`) | UDF/VIR raporları ve kullanıcı geri bildirimleri |

**Canlı adres:** http://84.8.250.171:5000/

---

## 2. Flask Nedir ve Bu Projede Nasıl Kullanılıyor?

**Flask**, Python ile web uygulaması yazmak için kullanılan hafif bir çatıdır (framework).

Bu projede Flask şunları yapar:

1. **URL dinler** — örneğin `/` ana sayfa, `/karar/959` UDF detay sayfası
2. **Veritabanından veri çeker** — PostgreSQL tablolarını okur
3. **HTML üretir** — Python string'leri içindeki şablonları (Jinja2) doldurur
4. **Form gönderimlerini işler** — Onayla / Override butonları POST isteği gönderir

### Temel yapı (`qaisu_app.py`)

```python
from flask import Flask, render_template_string, request

app = Flask(__name__)

@app.route('/')                    # Ana sayfa
def index():
    ...

@app.route('/karar/<int:udf_no>')  # UDF detay
def karar_detay(udf_no):
    ...

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

- `@app.route(...)` → Hangi URL'de hangi fonksiyonun çalışacağını tanımlar
- `render_template_string(...)` → HTML şablonunu verilerle birleştirip tarayıcıya gönderir
- `app.run(...)` → Sunucuyu başlatır; `0.0.0.0` dışarıdan erişime izin verir

---

## 3. Dosya Yapısı

```
Desktop/
├── qaisu_app_v9.py          ← Güncel Flask uygulaması (Mac'te geliştirme)
├── qaisu_model_v4.py        ← Model ve süreç önerisi fonksiyonları
├── sunucuya_yukle.sh        ← Mac'ten sunucuya otomatik yükleme scripti
└── QAISU_KULLANIM_KILAVUZU.md

Sunucu (/home/ubuntu/qaisu/):
├── qaisu_app.py             ← Canlıda çalışan Flask dosyası
└── qaisu_model.py           ← Canlıda kullanılan model dosyası
```

> **Önemli:** Mac'teki `qaisu_app_v9.py` ile sunucudaki `qaisu_app.py` aynı dosya değildir. Değişiklikleri görmek için sunucuya yüklemeniz gerekir.

---

## 4. Flask Uygulamasını Açma

### A) Sunucuda (canlı sistem — sizin kullandığınız)

**1. SSH ile sunucuya bağlanın:**

```bash
ssh -i ~/Desktop/ssh-key-2026-03-18.key ubuntu@84.8.250.171
```

**2. Uygulama klasörüne gidin ve başlatın:**

```bash
cd /home/ubuntu/qaisu
python3 qaisu_app.py
```

**Başarılı çıktı:**

```
 * Serving Flask app 'qaisu_app'
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://10.0.0.194:5000
```

**3. Tarayıcıda açın:**

```
http://84.8.250.171:5000/
```

> Terminali kapatırsanız veya `Ctrl+C` yaparsanız site kapanır. SSH bağlantısı kesilince de kapanmaması için arka planda çalıştırabilirsiniz:

```bash
cd /home/ubuntu/qaisu
nohup python3 qaisu_app.py > /tmp/qaisu_app.log 2>&1 &
```

---

### B) Mac'te yerel test (isteğe bağlı)

```bash
cd ~/Desktop
pip3 install flask sqlalchemy pandas psycopg2-binary
python3 qaisu_app_v9.py
```

Tarayıcı: http://localhost:5000/

> Yerel çalıştırmak için PostgreSQL'in Mac'te de ayakta olması gerekir.

---

## 5. Mac'ten Sunucuya Güncelleme Yükleme

Kodda değişiklik yaptıktan sonra canlı siteye yansıması için:

```bash
bash ~/Desktop/sunucuya_yukle.sh
```

Bu script:
1. `qaisu_app_v9.py` → sunucuya `qaisu_app.py` olarak kopyalar
2. `qaisu_model_v4.py` → sunucuya `qaisu_model.py` olarak kopyalar
3. Eski Flask sürecini durdurur, yeniden başlatır

Tarayıcıda **Cmd + Shift + R** ile sert yenileme yapın.

---

## 6. Sayfa ve URL Yapısı

| URL | Sayfa | Açıklama |
|-----|-------|----------|
| `/` | Ana sayfa | UDF ve VIR listeleri, istatistik kartları |
| `/karar/218` | UDF detay | Tek bir UDF kaydının incelemesi |
| `/vir/VIR-187529` | VIR detay | Tek bir VIR kaydının incelemesi |
| `POST /onayla/218` | Onay | Sistem kararını kabul eder |
| `POST /override/218` | Override | Mühendis farklı karar verir |
| `POST /vir/onayla/...` | VIR onay | VIR sistem kararını kabul |
| `POST /vir/override/...` | VIR override | VIR için farklı karar |

---

## 7. Ana Sayfa Kullanımı

Ana sayfada 6 istatistik kartı görünür:

| Kart | Anlamı |
|------|--------|
| UDF Toplam | Toplam UDF kayıt sayısı |
| UDF Uyuşmazlık | İnsan kararı ile sistem kararı uyuşmayan UDF |
| VIR Toplam | Toplam VIR kayıt sayısı |
| VIR Uyuşmazlık | Uyuşmayan VIR kayıtları |
| Onaylanan | Sistem kararı onaylanmış kayıtlar (`feedback_log`) |
| OVERRIDE | Mühendisin sistemi ezip farklı karar verdiği kayıtlar |

**UDF Kararları** ve **VIR Kararları** sekmeleri arasında geçiş yapabilirsiniz. **İncele** butonu ilgili detay sayfasına götürür.

---

## 8. UDF Detay Sayfası (`/karar/<udf_no>`)

### Sekmeler

1. **Üretim Hattı (UDF) Bilgileri** — Parça, iş emri, hata türü, anomali skoru vb.
2. **Tedarikçi Sapma (VIR) Geçmişi** — Aynı stok koduna ait geçmiş VIR kayıtları

> Sekme geçişi `switchVakaTab()` JavaScript fonksiyonu ile çalışır. Bu fonksiyon sunucu dosyasında yoksa sekme tıklanmaz (konsolda `switchVakaTab is not defined` hatası görünür).

### Karar bölümü

| Alan | Açıklama |
|------|----------|
| Geçmiş İnsan Kararı | Veritabanındaki eski insan kararı |
| Sistem Kararı | Optimizasyon modelinin önerdiği karar |
| Süreç Önerisi | Yapılması gereken aksiyon (SCAR, CAPA, izleme vb.) |
| Karar Gerekçesi | Maliyet, tedarikçi ve anomali risk yüzdeleri |

### Butonlar

- **SİSTEM KARARINI ONAYLA** → Sistem kararını kabul eder, `feedback_log` tablosuna yazar
- **KARARI DEĞİŞTİR (OVERRIDE)** → Kendi kararınızı seçip açıklama yazarsınız

---

## 9. Süreç Önerisi Terimleri

| Terim | Anlamı |
|-------|--------|
| **SCAR** | Tedarikçiye açılan Düzeltici Faaliyet kaydı |
| **CAPA** | Düzeltici / Önleyici Faaliyet (üretim kaynaklı hatalar) |
| **Islah** | Parçanın hurda yerine düzeltilerek kullanılması |
| **RTV** | Return to Vendor — parçanın tedarikçiye iadesi |
| **RU** | Koşullu kullanım / sınırlı kabul |
| **OGK** | Olağanüstü / geçici kabul |

**Örnek:** `SCAR + İZLE` → Tedarikçiye SCAR açın ve parçayı ıslah sonrası tekrar muayene edin.

---

## 10. Veritabanı Bağlantısı

Flask uygulaması şu bağlantıyı kullanır:

```
postgresql+psycopg2://qaisu:qaisu_pass@localhost:5432/kalite_db
```

Okunan tablolar:

| Tablo | İçerik |
|-------|--------|
| `superset_udf_karar_raporu` | UDF karar raporları |
| `superset_vir_risk_raporu` | VIR risk raporları |
| `feedback_log` | Onay / override kayıtları (Flask oluşturur) |

Model scripti (`qaisu_model.py`) çalıştırıldığında UDF/VIR tabloları güncellenir; Flask bu tabloları okur.

---

## 11. Sık Karşılaşılan Sorunlar

### Site açılmıyor (`ERR_CONNECTION_REFUSED`)

Flask çalışmıyor demektir. SSH'de:

```bash
cd /home/ubuntu/qaisu
python3 qaisu_app.py
```

---

### `Address already in use` (Port 5000 dolu)

Eski süreç hâlâ çalışıyor:

```bash
pkill -f "python3 qaisu_app.py"
sleep 2
cd /home/ubuntu/qaisu
python3 qaisu_app.py
```

---

### Sekme çalışmıyor (`switchVakaTab is not defined`)

Sunucudaki `qaisu_app.py` güncel değil. Mac'ten:

```bash
bash ~/Desktop/sunucuya_yukle.sh
```

Kontrol (SSH'de):

```bash
grep "function switchVakaTab" /home/ubuntu/qaisu/qaisu_app.py
```

Satır görünmeli.

---

### Değişiklikler görünmüyor

1. Dosyayı sunucuya yüklediniz mi?
2. Flask'ı yeniden başlattınız mı?
3. Tarayıcıda **Cmd + Shift + R** yaptınız mı?

---

### `favicon.ico 404`

Site ikonu eksik; işlevselliği etkilemez, yok sayabilirsiniz.

---

## 12. Geliştirme Akışı (Özet)

```
1. Mac'te qaisu_app_v9.py dosyasını düzenle
2. bash ~/Desktop/sunucuya_yukle.sh  → sunucuya yükle
3. Tarayıcıda Cmd + Shift + R          → sayfayı yenile
4. http://84.8.250.171:5000/           → test et
```

---

## 13. Hızlı Komut Referansı

| Ne yapmak istiyorsunuz? | Komut |
|-------------------------|-------|
| Sunucuya bağlan | `ssh -i ~/Desktop/ssh-key-2026-03-18.key ubuntu@84.8.250.171` |
| Flask başlat | `cd /home/ubuntu/qaisu && python3 qaisu_app.py` |
| Flask durdur | `Ctrl + C` veya `pkill -f "python3 qaisu_app.py"` |
| Güncelleme yükle | `bash ~/Desktop/sunucuya_yukle.sh` |
| Log kontrol | `tail -20 /tmp/qaisu_app.log` |
| Çalışıyor mu? | `pgrep -af qaisu_app.py` |

---

*Son güncelleme: Haziran 2026 — QAISU v9*
