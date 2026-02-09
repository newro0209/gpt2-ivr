---
# Fill in the fields below to create a basic custom agent for your repository.
# The Copilot CLI can be used for local testing: https://gh.io/customagents/cli
# To make this agent available, merge this file into the default repository branch.
# For format details, see: https://gh.io/customagents/config

name: 1인 개발을 함께 하는 에이전트
description: 함께 연구하며 개발한다. 모든 응답은 한국어로 대답해야 한다. 
---

# My Agent

## 프로젝트 구조 및 모듈 구성

- 프로젝트 목적을 `Tokenizer Model Migration + IVR`로 유지한다.
- 소스 코드는 `src/gpt2_ivr/` 표준 구조로 배치한다.
- 파이프라인 제어 코드는 `src/gpt2_ivr/cli.py`와 `src/gpt2_ivr/commands/`에 배치한다.
- 분석 로직은 `src/gpt2_ivr/analysis/`에 배치한다.
- 모든 산출물은 `artifacts/` 아래에 저장한다:
  - 분석 산출물: `artifacts/analysis/reports/`
  - 토크나이저 산출물: `artifacts/tokenizers/{original,distilled_unigram,remapped}/`
  - 코퍼스: `artifacts/corpora/{raw,cleaned}/`
  - 임베딩: `artifacts/embeddings/`
  - 학습 체크포인트: `artifacts/training/`

## 빌드, 테스트, 개발 명령어

- 환경 동기화는 `uv sync`를 사용한다.
- 전체 파이프라인은 아래 순서로 실행한다.
  - `uv run ivr analyze`
  - `uv run ivr distill-tokenizer`
  - `uv run ivr select`
  - `uv run ivr remap`
  - `uv run ivr align`
  - `uv run ivr train`
- `distill-tokenizer` 단계는 `remap` 이전에 반드시 수행한다.
- 엔트리 포인트 실행은 개별 스크립트 직접 호출 대신 `uv run ivr <command>` 형식을 기본으로 사용한다.

## 코딩 스타일 및 네이밍 규칙

- Python 들여쓰기는 공백 4칸을 사용한다.
- 함수명은 `snake_case`를 사용하고 목적이 드러나는 이름을 사용한다.
- CLI 오케스트레이션 로직은 `src/gpt2_ivr/cli.py`와 `src/gpt2_ivr/commands/`로 분리한다.
- 분석 코드와 실행 제어 코드를 혼합하지 않는다.
- 토크나이저/임베딩/학습 단계 간 입출력 경로를 명시적으로 유지한다.
- PEP 규칙을 엄격히 준수한다(PEP 8, PEP 257, PEP 484, PEP 526, PEP 544 포함).
- 코드 포매팅은 Black Formatter를 사용한다.
- 포매팅 검증 또는 적용은 `uv run black .` 명령을 기준으로 수행한다.
- 타입 힌트 작성 시 `list`, `dict`, `str` 등 내장 제네릭 타입을 우선 사용한다.
- Python 코드는 항상 Pythonic한 방식으로 작성한다.
- 짧더라도 복잡한 로직에는 의도를 설명하는 주석을 추가한다.
- 병렬 처리 가능 구간에서는 `concurrent.futures` 활용을 우선 검토한다.
- 진행 상태 추적을 위해 로깅을 기본 적용한다.
- 시간이 오래 걸리는 작업에는 프로그래스바를 적용해 실행 가시성을 확보한다.
- 주석, 문서, 로그 메시지는 한국어로 작성한다.
- 가독성 개선 목적의 이모지 사용을 허용한다.

## 정적 분석 및 품질 기준

- Pyright 기반 정적 분석에서 타입 에러 0건을 유지한다.
- 변경 제출 전 정적 분석을 수행하고 결과를 확인한다.
- 타입 안전성 저하를 유발하는 임시 우회(`type: ignore` 남용 등)를 지양한다.
- 마크다운 파일 작성 또는 수정 시 마크다운 린트 검사를 수행한다.
- 마크다운 린트 검사는 `npx markdownlint-cli <파일명>` 명령으로 실행한다.
- 린트 규칙 위반 사항이 없도록 수정하거나, 필요시 `.markdownlint.json`에서 규칙을 조정한다.
- 주요 마크다운 파일: `README.md`, `AGENTS.md`, `GEMINI.md` 등

## 테스트 가이드라인

- 현재 파이프라인의 1차 검증은 단계별 산출물 확인으로 수행한다.
- 아래 핵심 산출물 생성을 실행 검증 기준으로 사용한다.
  - `artifacts/analysis/reports/bpe_token_id_sequences.txt`
  - `artifacts/analysis/reports/replacement_candidates.csv`
  - `artifacts/tokenizers/distilled_unigram/`
  - `artifacts/tokenizers/remapped/`
- 테스트 프레임워크 도입 시 테스트 코드는 `tests/`에 배치하고 파일명은 `test_*.py` 규칙을 따른다.
- 자동화 테스트 명령어가 확정되면 본 문서와 `README.md`에 동시에 반영한다.

## 커밋 및 Pull Request 가이드라인

- 커밋 메시지는 Gitmoji를 포함해 작성한다.
- 커밋 제목과 본문은 한국어로 작성한다.
- 한 커밋에는 한 단계의 논리적 변경만 포함한다.
- PR에는 다음 항목을 포함한다:
  - 변경 사항과 연구/파이프라인 목적상 이유
  - 영향받는 단계(`analyze`, `distill-tokenizer`, `select`, `remap`, `align`, `train`) 명시
  - 생성/변경된 산출물 경로 목록
  - 로컬 실행 명령과 검증 결과(또는 미실행 사유)

## 설정 및 보안 참고사항

- 비밀값은 저장소에 커밋하지 않는다.
- 환경 변수 또는 로컬 `.env` 파일로 비밀값을 관리한다(`.env` 도입 시 `.gitignore`에 추가).
- 기본 실행 환경은 Python `3.13~3.14`, CUDA `13.0`, PyTorch `2.10` 기준으로 관리한다.
- Tokenizer 구현은 Hugging Face `tokenizers` 기반 Unigram을 사용한다.
- 학습 실행은 Hugging Face `accelerate` 기반으로 유지한다.
- 설정/경로/명령 변경 시 `README.md`와 본 가이드라인을 동시에 갱신한다.
