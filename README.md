# gpt2-ivr

> **BPE → Unigram Tokenizer Distillation 이후 IVR를 수행하는 연구/실험 표준 구조**

이 저장소는 단순 파인튜닝 프로젝트가 아닙니다.
목표는 **사전학습된 GPT‑2의 토크나이저 모델을 안전하게 교체한 뒤**,
그 위에서 **Vocabulary Reassignment(IVR)** 를 수행하는 재현성 있는 연구 파이프라인을 구축하는 것입니다.

---

## 🎯 프로젝트 목표

다음 두 단계를 **연속된 하나의 파이프라인**으로 수행합니다.

1. GPT‑2의 **BPE 토크나이저를 Unigram 토크나이저로 교체**
2. Unigram 토크나이저가 **BPE와 완전히 동일한**

   - 입력 → token id 시퀀스
   - token id 시퀀스 → 디코딩 결과
     를 만들도록 학습 (**Tokenizer Distillation**)
3. 그 위에서 **IVR(In‑place Vocabulary Reassignment)** 수행
4. embedding 재정렬 후 미세조정

즉, 이 프로젝트의 핵심은:

> **Tokenizer Model Migration + IVR**

입니다.

---

## 🗂️ 디렉토리 구조

```text
gpt2-ivr/
├─ README.md                    # 프로젝트 개요, 파이프라인, 실행 방법 문서
├─ pyproject.toml               # 패키지 메타데이터, 의존성, 엔트리 포인트 설정
├─ uv.lock                      # uv 의존성 락파일(재현 가능한 환경 고정)
│
├─ scripts/                     # 파이프라인 외 보조 유틸 스크립트
│   ├─ set_internal_pypi_index.*    # 내부 PyPI 인덱스 설정
│   └─ unset_internal_pypi_index.*  # 내부 PyPI 인덱스 해제
│
├─ artifacts/                   # 파이프라인 산출물 저장 루트
│   ├─ corpora/                 # 코퍼스 데이터
│   │   ├─ raw/                 # 원본 수집 데이터
│   │   └─ cleaned/             # 전처리/정제 완료 데이터
│   ├─ tokenizers/              # 토크나이저 산출물
│   │   ├─ original/            # 원본 GPT-2 토크나이저
│   │   ├─ distilled_unigram/   # Distillation 완료 Unigram 토크나이저
│   │   └─ remapped/            # IVR 적용 후 토크나이저
│   ├─ analysis/                # 분석 산출물
│   │   └─ reports/             # 분석 리포트
│   ├─ embeddings/              # 임베딩 산출물
│   ├─ logs/                    # 실행 로그 파일
│   └─ training/                # 학습 체크포인트 및 로그
│
└─ src/                         # 패키지 소스 루트
    └─ gpt2_ivr/                # 메인 패키지
        ├─ __init__.py          # 패키지 초기화
        ├─ cli.py               # `uv run ivr ...` CLI 엔트리 포인트
        │
        ├─ commands/            # Command 패턴 구현
        │   ├─ __init__.py
        │   ├─ base.py          # Command 추상 클래스
        │   ├─ init_command.py
        │   ├─ analyze_command.py
        │   ├─ distill_command.py
        │   └─ select_command.py
        │
        ├─ analysis/            # 분석 및 후보 선정 로직
        │   ├─ __init__.py
        │   ├─ token_frequency.py       # 토큰 빈도 통계 계산
        │   ├─ candidate_selection.py   # IVR 교체 후보 토큰 선정
        │   └─ bpe_corpus_export.py     # GPT-2 BPE 기준 토큰 시퀀스 추출
        │
        ├─ tokenizer/           # 토크나이저 로직
        │   ├─ __init__.py
        │   ├─ distill.py       # Unigram distillation 핵심 로직
        │   ├─ validate.py      # distillation encode/decode 동일성 검증
        │   └─ remap_rules.yaml # 토큰 재할당 규칙 정의
        │
        ├─ embedding/           # 임베딩 추출/재배치/초기화 로직
        │   ├─ __init__.py
        │   ├─ extract.py       # 기존 모델 임베딩 추출
        │   ├─ reorder.py       # remap 규칙 기준 임베딩 재정렬
        │   └─ init_new.py      # 신규 토큰 임베딩 초기화
        │
        ├─ training/            # 학습 설정 및 학습 실행 코드
        │   ├─ __init__.py
        │   ├─ accelerate_config.yaml   # accelerate 실행 환경/분산 설정
        │   ├─ sft_config.yaml          # 미세조정 하이퍼파라미터/런타임 설정
        │   └─ train.py                 # accelerate 기반 학습 실행
        │
        └─ utils/               # 공통 유틸리티
            ├─ __init__.py
            └─ logging_config.py
```

---

## ▶️ 실행 파이프라인 (엔트리 포인트)

모든 단계는 엔트리 포인트를 통해 실행합니다.

```bash
uv run ivr init
uv run ivr analyze
uv run ivr distill-tokenizer
uv run ivr select
uv run ivr remap
uv run ivr align
uv run ivr train
```

Tokenizer Distillation 단계는 **반드시 IVR 이전**에 수행됩니다.

---

## 🧠 Tokenizer Distillation (핵심 개념)

이 단계의 목적은 **토큰이나 id를 바꾸는 것이 아닙니다.**

> **토크나이저 “모델”만 BPE → Unigram으로 교체**하면서
> 모델이 보는 token id 시퀀스를 완전히 동일하게 유지하는 것

### Distillation 이후 만족해야 하는 조건

| 항목             | 상태                      |
|----------------|------------------------|
| id ↔ token     | GPT‑2와 동일              |
| encode 결과      | GPT‑2와 동일              |
| decode 결과      | GPT‑2와 동일              |
| tokenizer 모델   | Unigram (merges 없음)    |

### 방법

1. GPT‑2 BPE로 코퍼스를 전부 토큰화하여 **정답 token id 시퀀스** 생성
2. 이 시퀀스를 Unigram 학습의 label로 사용
3. vocab size = 50257 유지

결과적으로 모델은 **토크나이저가 바뀐 것을 인지하지 못합니다.**

---

## 🔧 IVR 단계

Distilled Unigram 위에서 저빈도 토큰을 도메인 고빈도 토큰으로 교체합니다.

```text
replacement_candidates.csv
        ↓
remap_rules.yaml
        ↓
src/gpt2_ivr/embedding/reorder.py
        ↓
src/gpt2_ivr/training/train.py
```

Distillation은 “안 깨지게 옮기는 단계”,
IVR은 “토큰 표현력을 개선하는 단계”입니다.

---

## 📁 분석 산출물 (연구 자산)

```text
artifacts/analysis/reports/
├─ token_frequency.parquet
├─ replacement_candidates.csv
├─ bpe_token_id_sequences.txt
└─ selection_log.md
```

이 파일들은 코드보다 더 중요한 **연구 기록**입니다.

---

## 🧩 역할 분리 원칙

| 위치                         | 역할                                          |
|------------------------------|----------------------------------------------|
| `src/gpt2_ivr/cli.py`        | CLI 엔트리 포인트                               |
| `src/gpt2_ivr/commands/`     | Command 패턴 구현 (파이프라인 오케스트레이션)         |
| `src/gpt2_ivr/analysis/`     | 분석 로직 (Research Library)                   |
| `src/gpt2_ivr/tokenizer/`    | 토크나이저 로직                                 |
| `src/gpt2_ivr/embedding/`    | 임베딩 추출/재배치 로직                           |
| `src/gpt2_ivr/training/`     | 학습 설정 및 실행 로직                            |
| `artifacts/*`                | 토크나이저/분석/임베딩/학습 산출물                   |
| `scripts/*`                  | 파이프라인 외 보조 유틸리티 스크립트                 |

---

## 🏗️ 아키텍처 계층 구조

이 프로젝트는 **Layered Architecture** 패턴을 따라 관심사를 명확히 분리합니다.

### 1️⃣ 프레젠테이션 계층 (Presentation Layer)

- **위치**: `cli.py`
- **책임**: 사용자 인터페이스(CLI) 제공
- **역할**:
  - 사용자 입력을 받아 적절한 Command로 라우팅
  - argparse 기반 명령행 인터페이스 제공
  - 배너 출력 및 로깅 초기화

### 2️⃣ 애플리케이션 계층 (Application Layer)

- **위치**: `commands/`
- **책임**: 명령 오케스트레이션 및 제어 흐름
- **역할**:
  - 도메인 로직을 조합하여 비즈니스 유스케이스 구현
  - 입출력 경로 관리 및 파라미터 전달
  - Command 패턴을 통한 실행 단위 캡슐화

### 3️⃣ 도메인 계층 (Domain Layer)

- **위치**: `analysis/`, `tokenizer/`, `embedding/`, `training/`
- **책임**: 핵심 비즈니스 로직 및 알고리즘 구현
- **역할**:
  - CLI/Command와 독립적으로 재사용 가능한 로직
  - 토큰 분석, 토크나이저 증류, 임베딩 처리, 모델 학습 등 핵심 기능
  - 연구 및 실험의 핵심 자산

### 4️⃣ 유틸리티 계층 (Infrastructure/Utility Layer)

- **위치**: `utils/`
- **책임**: 공통 유틸리티 및 인프라 지원
- **역할**:
  - 로깅 설정 등 횡단 관심사(Cross-cutting Concerns) 처리
  - 모든 계층에서 공통으로 사용하는 기능 제공

### 계층 간 의존성 규칙

```text
프레젠테이션 계층 (cli.py)
        ↓
애플리케이션 계층 (commands/)
        ↓
도메인 계층 (analysis/, tokenizer/, embedding/, training/)
        ↓
유틸리티 계층 (utils/)
```

- **단방향 의존성**: 상위 계층은 하위 계층에만 의존
- **도메인 독립성**: 도메인 계층은 CLI/Command 계층을 알지 못함
- **재사용성**: 각 계층은 독립적으로 테스트 및 재사용 가능

---

## 📂 중앙화된 경로 상수 관리

모든 artifacts 경로는 `src/gpt2_ivr/constants.py`에서 중앙 관리됩니다.

### 주요 경로 상수

```python
from gpt2_ivr.constants import (
    # 코퍼스 경로
    CORPORA_CLEANED_DIR,           # artifacts/corpora/cleaned
    
    # 토크나이저 경로
    TOKENIZER_ORIGINAL_DIR,        # artifacts/tokenizers/original
    TOKENIZER_DISTILLED_UNIGRAM_DIR,  # artifacts/tokenizers/distilled_unigram
    TOKENIZER_REMAPPED_DIR,        # artifacts/tokenizers/remapped
    
    # 분석 산출물 경로
    BPE_TOKEN_ID_SEQUENCES_FILE,   # artifacts/analysis/reports/bpe_token_id_sequences.txt
    TOKEN_FREQUENCY_FILE,          # artifacts/analysis/reports/token_frequency.parquet
    REPLACEMENT_CANDIDATES_FILE,   # artifacts/analysis/reports/replacement_candidates.csv
    SELECTION_LOG_FILE,            # artifacts/analysis/reports/selection_log.md
    
    # 로그 및 학습 경로
    LOGS_DIR,                      # artifacts/logs
    TRAINING_CHECKPOINT_DIR,       # artifacts/training/sft_checkpoint
)
```

### 장점

- **일관성**: 모든 코드가 동일한 경로 상수를 참조
- **유지보수성**: 경로 변경 시 한 곳만 수정
- **가독성**: 경로의 의미가 명확한 상수명으로 표현
- **타입 안전성**: Path 객체로 타입 체크 가능

---

## 🧰 환경 및 도구

| 항목           | 스택                                      |
|--------------|-----------------------------------------|
| 환경 관리        | uv                                      |
| Python       | 3.13 ~ 3.14                             |
| Tokenizer    | Hugging Face `tokenizers` (Unigram)     |
| Training     | Hugging Face `accelerate`               |
| Base Model   | `openai-community/gpt2`                 |
| CUDA         | 13.0                                    |
| PyTorch      | 2.10                                    |

---

## ✅ 이 구조가 보장하는 것

- BPE → Unigram 안전 이식
- 그 위에서 IVR 수행
- 분석 결과의 파일 기반 축적
- 재현 가능한 엔드투엔드 파이프라인

---

## 🚀 Quick Start

### 1️⃣ 환경 준비

```bash
uv sync
```

> Python 3.13~3.14, CUDA 13.0, PyTorch 2.10 환경을 전제로 합니다.

---

### 2️⃣ 코퍼스 준비

```text
artifacts/corpora/raw/     # 원본 데이터 수집
artifacts/corpora/cleaned/ # 정제 완료 데이터
```

---

### 3️⃣ 모델 및 토크나이저 초기화

```bash
uv run ivr init
```

- Hugging Face Hub에서 GPT-2 토크나이저와 모델 설정을 다운로드
- 산출물: `artifacts/tokenizers/original/`
- `--force` 옵션으로 기존 파일이 있어도 다시 다운로드 가능

---

### 4️⃣ BPE 토큰 시퀀스 생성

```bash
uv run ivr analyze
```

- GPT‑2 BPE 기준 token id 시퀀스를 생성
- 산출물: `artifacts/analysis/reports/bpe_token_id_sequences.txt`

---

### 5️⃣ Tokenizer Distillation (BPE → Unigram)

```bash
uv run ivr distill-tokenizer
```

- BPE와 동일한 encode/decode를 만드는 Unigram tokenizer 생성
- 산출물: `artifacts/tokenizers/distilled_unigram/`

---

### 6️⃣ IVR 대상 토큰 선정

```bash
uv run ivr select
```

- 저빈도 토큰 분석
- 산출물: `artifacts/analysis/reports/replacement_candidates.csv`

---

### 7️⃣ 토큰 교체 및 tokenizer 생성

```bash
uv run ivr remap
```

- IVR 적용 tokenizer 생성
- 산출물: `artifacts/tokenizers/remapped/`

---

### 8️⃣ Embedding 재정렬

```bash
uv run ivr align
```

- GPT‑2 모델에서 토큰 임베딩(wte)과 위치 임베딩(wpe) 추출
- Remap 규칙에 따라 임베딩 재정렬
- 신규 추가된 토큰에 대한 임베딩 초기화
- 산출물: `artifacts/embeddings/`

#### 주요 옵션

```bash
uv run ivr align \
  --model-name openai-community/gpt2 \
  --original-tokenizer-dir artifacts/tokenizers/original \
  --remapped-tokenizer-dir artifacts/tokenizers/remapped \
  --remap-rules-path src/gpt2_ivr/tokenizer/remap_rules.yaml \
  --embeddings-output-dir artifacts/embeddings \
  --init-strategy mean
```

- `--init-strategy`: 신규 토큰 임베딩 초기화 전략
  - `mean`: 기존 임베딩 평균값 사용 (기본값)
  - `random`: 정규분포 랜덤 초기화
  - `zeros`: 0으로 초기화

#### 처리 단계

1. **Extract**: 원본 GPT-2 모델에서 임베딩 추출
2. **Reorder**: Remap 규칙에 따라 토큰 임베딩 재배치
3. **Initialize**: 신규 토큰 임베딩 초기화

#### 산출물

- `artifacts/embeddings/original_wte.pt` - 원본 토큰 임베딩
- `artifacts/embeddings/original_wpe.pt` - 원본 위치 임베딩
- `artifacts/embeddings/aligned_wte.pt` - 재정렬된 토큰 임베딩
- `artifacts/embeddings/final_wte.pt` - 최종 임베딩 (초기화 완료)
- `artifacts/embeddings/*.json` - 각 단계별 메타데이터

---

### 9️⃣ 미세조정

```bash
uv run ivr train
```

- accelerate 기반 학습 수행
