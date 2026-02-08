$ErrorActionPreference = "Stop"

$gatewayIp = (Get-NetIPAddress -AddressFamily IPv4 |
  Where-Object {
    $_.InterfaceAlias -match 'vEthernet' -and
    $_.InterfaceAlias -match 'WSL' -and
    $_.IPAddress -ne "0.0.0.0"
  } |
  Select-Object -First 1 -ExpandProperty IPAddress)

if (-not $gatewayIp) {
  Write-Error "vEthernet 및 WSL 문구가 포함된 인터페이스의 IPv4 주소를 찾지 못했습니다."
  exit 1
}
$indexUrl = "http://$gatewayIp`:3141/root/pypi/+simple/"
$fallbackUrl = "https://pypi.org/simple"

python -m pip config set global.index-url "$indexUrl"
python -m pip config set global.extra-index-url "$fallbackUrl"

Write-Host "사내 패키지 서버 설정 완료: $indexUrl"
Write-Host "공식 PyPI 폴백: $fallbackUrl"
