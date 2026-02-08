$ErrorActionPreference = "Stop"

try {
  python -m pip config unset global.index-url | Out-Null
} catch {
  # 설정이 없을 수도 있으므로 무시
}

try {
  python -m pip config unset global.extra-index-url | Out-Null
} catch {
  # 설정이 없을 수도 있으므로 무시
}

Write-Host "사내 패키지 서버 설정 해제 완료"
