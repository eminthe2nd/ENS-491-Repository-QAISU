# QAISU — Kalite Karar Destek Sistemi

Flask tabanlı kalite karar destek uygulaması. UDF ve VIR kayıtlarını listeler, sistem kararı ile insan kararını karşılaştırır, onay/override işlemlerini kaydeder.

## Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `qaisu_app.py` | Flask web uygulaması |
| `qaisu_model.py` | Model ve süreç önerisi fonksiyonları |
| `QAISU_KULLANIM_KILAVUZU.md` | Detaylı kullanım kılavuzu |
| `sunucuya_yukle.sh` | Sunucuya deployment scripti |

## Hızlı Başlangıç (Sunucu)

```bash
cd /home/ubuntu/qaisu
python3 qaisu_app.py
```

Tarayıcı: http://84.8.250.171:5000/

## Gereksinimler

- Python 3
- PostgreSQL (`kalite_db`)
- `flask`, `sqlalchemy`, `pandas`, `psycopg2-binary`

Detaylı kurulum ve kullanım için `QAISU_KULLANIM_KILAVUZU.md` dosyasına bakın.
