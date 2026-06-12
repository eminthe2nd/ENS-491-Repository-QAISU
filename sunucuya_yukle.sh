#!/bin/bash
set -e

KEY="$HOME/Desktop/ssh-key-2026-03-18.key"
LOCAL_DIR="$HOME/Desktop/QAISU_SonSurum"
LOCAL_APP="$LOCAL_DIR/qaisu_app.py"
LOCAL_MODEL="$LOCAL_DIR/qaisu_model.py"
REMOTE_APP="ubuntu@84.8.250.171:/home/ubuntu/qaisu/qaisu_app.py"
REMOTE_MODEL="ubuntu@84.8.250.171:/home/ubuntu/qaisu/qaisu_model.py"

echo "Güncel dosyalar sunucuya kopyalanıyor..."
scp -i "$KEY" -o StrictHostKeyChecking=no "$LOCAL_APP" "$REMOTE_APP"
scp -i "$KEY" -o StrictHostKeyChecking=no "$LOCAL_MODEL" "$REMOTE_MODEL"

echo "Uygulama yeniden başlatılıyor..."
ssh -i "$KEY" -o StrictHostKeyChecking=no ubuntu@84.8.250.171 << 'EOF'
pkill -f "python3 qaisu_app.py" 2>/dev/null || true
sleep 1
cd /home/ubuntu/qaisu
nohup python3 qaisu_app.py > /tmp/qaisu_app.log 2>&1 &
sleep 2
echo "--- Sunucu durumu ---"
pgrep -af qaisu_app.py || echo "UYARI: uygulama başlamadı"
tail -3 /tmp/qaisu_app.log 2>/dev/null || true
EOF

echo ""
echo "Tamamlandı. Tarayıcıda açın:"
echo "http://84.8.250.171:5000/"
