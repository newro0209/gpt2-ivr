#!/usr/bin/env bash
set -euo pipefail

python -m pip config unset global.index-url || true
python -m pip config unset global.extra-index-url || true

echo "사내 패키지 서버 설정 해제 완료"
