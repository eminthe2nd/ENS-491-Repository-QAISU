#!/bin/bash
# GitHub'a push etmek icin Mac terminalinde calistirin:
#   bash ~/Desktop/QAISU_SonSurum/github_push.sh
#
# ONCE: GitHub davetini kabul edin (View invitation)
# ONCE: gh auth login  (gerekirse)

set -e
REPO_DIR="$HOME/Desktop/QAISU_SonSurum"
REMOTE="https://github.com/eminthe2nd/QAISU-Repo.git"

cd "$REPO_DIR"

echo "=== 1) Git hazirligi ==="

# Bozuk/yarim kalmis .git klasorunu temizle
if [ -d .git ] && ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "Bozuk .git klasoru siliniyor..."
  rm -rf .git
fi

if [ ! -d .git ]; then
  git init
  git branch -M main
fi

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REMOTE"
else
  git remote add origin "$REMOTE"
fi

echo "=== 2) Uzak repodan guncelleme cekiliyor ==="
git fetch origin main

if git rev-parse --verify origin/main >/dev/null 2>&1; then
  git pull origin main --allow-unrelated-histories --no-edit || true
fi

echo "=== 3) Dosyalar ekleniyor ==="
git add .gitignore README.md qaisu_app.py qaisu_model.py QAISU_KULLANIM_KILAVUZU.md sunucuya_yukle.sh sunucu_sekme_duzelt.sh github_push.sh

if git diff --cached --quiet; then
  echo "Yeni degisiklik yok."
else
  git commit -m "$(cat <<'EOF'
QAISU Flask uygulamasinin guncel surumu.

Flask web arayuzu, model dosyasi, kullanim kilavuzu ve deployment scriptleri eklendi.
EOF
)"
fi

echo "=== 4) GitHub'a push ==="
git push -u origin main

echo ""
echo "Tamamlandi: https://github.com/eminthe2nd/QAISU-Repo"
