#!/usr/bin/env bash
set -euo pipefail

gateway_ip=$(ip route | awk '/^default/ {print $3; exit}')
if [[ -z "${gateway_ip}" ]]; then
  echo "기본 게이트웨이를 찾지 못했습니다." >&2
  exit 1
fi

index_url="http://${gateway_ip}:3141/root/pypi/+simple/"
fallback_url="https://pypi.org/simple"

python -m pip config set global.index-url "${index_url}"
python -m pip config set global.extra-index-url "${fallback_url}"

echo "사내 패키지 서버 설정 완료: ${index_url}"
echo "공식 PyPI 폴백: ${fallback_url}"
