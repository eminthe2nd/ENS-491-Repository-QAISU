#!/bin/bash
# SSH oturumunda sunucuya bagliyken calistirin:
#   bash sunucu_sekme_duzelt.sh
#
# Mac'ten once dosyayi kopyalamak icin (yeni terminal sekmesi):
#   scp -i ~/Desktop/ssh-key-2026-03-18.key ~/Desktop/sunucu_sekme_duzelt.sh ubuntu@84.8.250.171:~/
#   ssh -i ~/Desktop/ssh-key-2026-03-18.key ubuntu@84.8.250.171
#   bash ~/sunucu_sekme_duzelt.sh

set -e
APP="/home/ubuntu/qaisu/qaisu_app.py"

python3 << PY
from pathlib import Path

path = Path("$APP")
text = path.read_text(encoding="utf-8")

script = """
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
"""

marker = '</div>\\n</body>\\n</html>\\n"""\\n\\n# ============================================================\\n# VIR KARAR DETAY SAYFASI'
replacement = '</div>\\n' + script + '\\n</body>\\n</html>\\n"""\\n\\n# ============================================================\\n# VIR KARAR DETAY SAYFASI'

if 'function switchVakaTab' in text:
    print('switchVakaTab zaten mevcut.')
elif marker in text:
    text = text.replace(marker, replacement, 1)
    path.write_text(text, encoding='utf-8')
    print('switchVakaTab basariyla eklendi.')
else:
    print('HATA: Dosya yapisi farkli. Mac terminalinden su komutu deneyin:')
    print('  bash ~/Desktop/sunucuya_yukle.sh')
    raise SystemExit(1)
PY

echo "Flask yeniden baslatiliyor..."
pkill -f "python3 qaisu_app.py" 2>/dev/null || true
sleep 1
cd /home/ubuntu/qaisu
nohup python3 qaisu_app.py > /tmp/qaisu_app.log 2>&1 &
sleep 2
pgrep -af qaisu_app.py || { echo "UYARI: Uygulama baslamadi. Log:"; tail -20 /tmp/qaisu_app.log; exit 1; }
echo ""
echo "Tamam. Tarayicida sayfayi yenileyin (Ctrl+F5):"
echo "http://84.8.250.171:5000/karar/959"
