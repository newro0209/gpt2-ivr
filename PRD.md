# GPT2-IVR 제품 요구사항 문서 (PRD)

**프로젝트명**: GPT2-IVR (GPT-2 In-place Vocabulary Reassignment)
**버전**: 1.0
**작성일**: 2026-02-10
**용도**: 토크나이저 모델 교체 및 어휘 재할당 연구 파이프라인

---

## 📌 제품 개요

### 프로젝트 정의

GPT2-IVR은 **사전학습된 GPT-2 모델의 토크나이저를 안전하게 교체**하고, **도메인별 어휘 재할당(In-place Vocabulary Reassignment, IVR)을 수행하는 재현 가능한 연구 파이프라인**입니다.

이 프로젝트는 단순한 파인튜닝이 아니라 **토크나이저 모델 마이그레이션과 어휘 최적화**의 두 단계를 통합적으로 처리합니다.

### 핵심 목표

1. **BPE → Unigram 토크나이저 마이그레이션**
   - GPT-2의 기존 BPE 토크나이저를 Unigram 토크나이저로 교체
   - 원본 토크나이저와 동일한 encode/decode 결과 유지

2. **토크나이저 증류(Tokenizer Distillation)**
   - Unigram 토크나이저가 BPE와 완전히 동일한 token id 시퀀스를 생성하도록 학습
   - 모델이 토크나이저 변경을 인지하지 못하도록 투명성 보장

3. **도메인별 어휘 재할당(IVR)**
   - 저빈도 토큰을 도메인 고빈도 토큰으로 교체
   - 도메인별 토큰 최적화

4. **임베딩 재정렬 및 미세조정**
   - 재할당된 토큰에 맞춰 모델 임베딩 재정렬
   - 신규 토큰 초기화 및 모델 미세조정

### 대상 사용자

- NLP 연구자
- 토크나이저 최적화에 관심이 있는 개발자
- 도메인별 언어모델 적응이 필요한 팀

### 주요 특징

- **투명한 토크나이저 교체**: 모델 동작 변화 최소화
- **재현 가능한 파이프라인**: 명확한 단계별 실행 구조
- **연구 기록 중심**: 각 단계별 분석 보고서 및 산출물 저장
- **확장 가능한 아키텍처**: 도메인별 커스터마이징 용이

---

## 🎯 핵심 기능

### 1단계: 코퍼스 준비 및 정제 (`init`)

**목적**: 다양한 형식의 원본 데이터를 통일된 텍스트 형식으로 정제

**입력**:
- `artifacts/corpora/raw/` 아래의 `.txt`, `.jsonl`, `.json` 파일

**처리**:
- JSON/JSONL 파일에서 텍스트 키 추출 (기본값: `text`)
- 인코딩 정규화 (기본값: UTF-8)
- 텍스트 정규화 (옵션)

**출력**:
- `artifacts/corpora/cleaned/` 아래의 정제된 `.txt` 파일

**옵션**:
- `--text-key`: JSON에서 텍스트 추출 키 지정
- `--encoding`: 텍스트 파일 인코딩 지정
- `--raw-corpora-dir`: 원본 데이터 디렉토리 지정
- `--cleaned-corpora-dir`: 출력 디렉토리 지정
- `--normalize-force`: 기존 정제본 덮어쓰기

---

### 2단계: 토큰 빈도 분석 (`analyze`)

**목적**: BPE 토크나이저로 코퍼스를 토큰화하고 토큰별 빈도 통계 생성

**입력**:
- 정제된 코퍼스 (`artifacts/corpora/cleaned/`)
- 원본 GPT-2 BPE 토크나이저 (`artifacts/tokenizers/original/`)

**처리**:
1. 클린 코퍼스의 모든 텍스트를 GPT-2 BPE로 토큰화
2. Token ID 시퀀스 저장
3. 각 토큰의 출현 빈도 집계
4. 병렬 처리를 통해 대용량 코퍼스 효율 처리

**출력**:
- `artifacts/analysis/reports/bpe_token_id_sequences.txt`
  - 토큰 ID 시퀀스 (한 라인당 한 문서)
- `artifacts/analysis/reports/token_frequency.parquet`
  - 토큰별 빈도 통계 (Parquet 형식)

**데이터 구조 (token_frequency.parquet)**:

```
| token_id | token_str | frequency |
|----------|-----------|-----------|
| 0        | !         | 1250      |
| 1        | "         | 3420      |
| ...      | ...       | ...       |
```

---

### 3단계: 토크나이저 증류 (`distill-tokenizer`)

**목적**: Unigram 토크나이저를 학습하여 BPE와 동일한 encode/decode 결과 생성

**핵심 개념**: Tokenizer Distillation

BPE 토크나이저의 동작을 모방하는 Unigram 토크나이저를 학습합니다.

| 항목 | 상태 |
|------|------|
| Token ID ↔ Token String | GPT-2와 동일 |
| Encode 결과 | GPT-2와 동일 |
| Decode 결과 | GPT-2와 동일 |
| 토크나이저 모델 | Unigram (Merges 없음) |

**입력**:
- 원본 GPT-2 BPE 토크나이저
- 정제된 코퍼스
- 원본 어휘 크기 (50257)

**처리**:
1. GPT-2 BPE로 코퍼스 전체 토큰화 → 정답 token ID 시퀀스 생성
2. Unigram 토크나이저를 이 시퀀스를 라벨로 하여 학습
3. 어휘 크기를 50257로 유지

**출력**:
- `artifacts/tokenizers/distilled_unigram/`
  - `tokenizer.json`: Unigram 토크나이저 정의
  - `special_tokens_map.json`: 특수 토큰 매핑
  - `vocab.txt`: 어휘 목록

**특징**:
- 모델이 토크나이저 변경을 감지할 수 없음
- 동일한 token id로 동일한 학습 유지

---

### 4단계: IVR 대상 토큰 선정 (`select`)

**목적**: 저빈도(희생) 토큰과 도메인 고빈도 토큰(신규) 매칭

**입력**:
- 토큰 빈도 정보 (`token_frequency.parquet`)
- BPE Token ID 시퀀스 (`bpe_token_id_sequences.txt`)

**처리**:
1. 저빈도 토큰 식별 (희생 후보)
2. 코퍼스의 연속 토큰 쌍(바이그램) 분석
3. 고빈도 바이그램 → 신규 토큰 후보로 선정
4. 희생 후보와 신규 후보 매칭

**출력**:
- `artifacts/analysis/reports/replacement_candidates.csv`

  ```
  sacrifice_id,sacrifice_token,new_token,sacrifice_freq,new_bigram_freq
  45632,<특정>,문제+해결,12,3450
  ...
  ```

- `artifacts/analysis/reports/selection_log.md`
  - 선정 기준 및 결과 요약

---

### 5단계: 토큰 재할당 및 토크나이저 생성 (`remap`)

**목적**: 선정된 토큰 쌍에 따라 Unigram 토크나이저 재할당

**입력**:
- 증류된 Unigram 토크나이저
- 교체 후보 목록 (`replacement_candidates.csv`)

**처리**:
1. `remap_rules.yaml` 생성
   - 각 희생 토큰과 신규 토큰의 매핑 규칙
2. Unigram 토크나이저의 어휘 재구성
3. 새로운 토크나이저 저장

**출력**:
- `src/gpt2_ivr/tokenizer/remap_rules.yaml`

  ```yaml
  45632: "문제해결"
  45633: "데이터분석"
  ...
  ```

- `artifacts/tokenizers/remapped/`
  - `tokenizer.json`: 재할당된 Unigram 토크나이저
  - 기타 메타데이터

---

### 6단계: 임베딩 추출 및 재정렬 (`align`)

**목적**: 원본 GPT-2 모델의 임베딩을 추출하고 재할당 규칙에 따라 재정렬

**입력**:
- 원본 GPT-2 모델 (`openai-community/gpt2`)
- 재할당 규칙 (`remap_rules.yaml`)
- 원본 및 재할당 토크나이저

**처리**:

#### 6.1 추출(Extract)

- GPT-2 모델에서 토큰 임베딩(wte) 추출
- 위치 임베딩(wpe) 추출
- 원본 크기: `(50257, 768)` (vocab_size × embedding_dim)

#### 6.2 재정렬(Reorder)

- 원본 임베딩을 재할당 규칙에 따라 재배치
- 기존 토큰은 위치 그대로 보존
- 희생 토큰의 임베딩 → 신규 토큰으로 이동

#### 6.3 초기화(Initialize)

- 신규 추가된 토큰의 임베딩 초기화
- 초기화 전략:
  - `mean`: 기존 임베딩의 평균값 사용 (기본값)
  - `random`: 정규분포 랜덤 초기화
  - `zeros`: 0으로 초기화

**옵션**:

```bash
uv run ivr align \
  --model-name openai-community/gpt2 \
  --original-tokenizer-dir artifacts/tokenizers/original \
  --remapped-tokenizer-dir artifacts/tokenizers/remapped \
  --remap-rules-path src/gpt2_ivr/tokenizer/remap_rules.yaml \
  --embeddings-output-dir artifacts/embeddings \
  --init-strategy mean
```

**출력**:
- `artifacts/embeddings/original_wte.pt`
  - 원본 토큰 임베딩 `(50257, 768)`

- `artifacts/embeddings/original_wpe.pt`
  - 원본 위치 임베딩 `(1024, 768)`

- `artifacts/embeddings/aligned_wte.pt`
  - 재정렬된 토큰 임베딩 (신규 토큰 포함)

- `artifacts/embeddings/final_wte.pt`
  - 최종 임베딩 (초기화 완료)

- `artifacts/embeddings/*.json`
  - 메타데이터 (원본 토큰 ID, 신규 토큰 ID, 초기화 전략 등)

---

### 7단계: 모델 미세조정 (`train`)

**목적**: 재할당된 토크나이저와 새로운 임베딩을 사용하여 모델 미세조정

**입력**:
- 재할당된 토크나이저
- 초기화된 임베딩
- 정제된 코퍼스

**처리**:
- Hugging Face Accelerate 기반 분산 학습
- 새로운 토큰과 재정렬된 임베딩의 최적화

**출력**:
- `artifacts/training/sft_checkpoint/`
  - 학습 체크포인트 및 모델 파라미터
  - 학습 로그 및 메트릭

**구현 상태**: 계획 단계

---

## ⚙️ 기술 스택 및 아키텍처

### 기술 스택

| 구성 요소 | 기술 | 버전 |
|---------|------|------|
| **언어** | Python | 3.13 ~ 3.14 |
| **환경 관리** | UV | 최신 |
| **토크나이저** | Hugging Face `tokenizers` | >= 0.22.2 |
| **모델 & 변환** | Hugging Face `transformers` | >= 5.1.0 |
| **데이터 처리** | `datasets`, `pandas`, `pyarrow` | 최신 |
| **학습** | PyTorch + Accelerate | >= 2.10.0 |
| **시각화 & CLI** | Rich | >= 14.3.2 |
| **설정 관리** | PyYAML | >= 6.0 |
| **정적 분석** | Pyright | >= 1.1.408 |
| **코드 포매팅** | Black | >= 26.1.0 |

### 아키텍처 개요

프로젝트는 **Layered Architecture** 패턴을 따릅니다.

```
┌─────────────────────────────────────────────┐
│    프레젠테이션 계층 (Presentation Layer)    │
│         src/gpt2_ivr/cli.py                 │
│    (CLI 인터페이스 & 로깅 설정)              │
└──────────────┬──────────────────────────────┘
               ↓
┌─────────────────────────────────────────────┐
│     애플리케이션 계층 (Application Layer)     │
│       src/gpt2_ivr/commands/                │
│  (파이프라인 오케스트레이션 & 제어 흐름)     │
└──────────────┬──────────────────────────────┘
               ↓
┌─────────────────────────────────────────────┐
│       도메인 계층 (Domain Layer)             │
│  • analysis/     (분석 로직)                 │
│  • tokenizer/    (토크나이저 로직)           │
│  • embedding/    (임베딩 처리)               │
│  • training/     (모델 학습)                 │
│  • corpus/       (코퍼스 정제)               │
│  (재사용 가능한 비즈니스 로직)               │
└─────────────────────────────────────────────┘
```

### 계층별 책임

#### 프레젠테이션 계층 (`cli.py`)

- **책임**: 사용자 인터페이스(CLI) 제공
- **역할**:
  - 사용자 입력을 받아 적절한 Command로 라우팅
  - argparse 기반 명령행 인터페이스
  - Rich 콘솔 출력 및 로깅 관리
  - ASCII 배너 및 프로그래스 표시

#### 애플리케이션 계층 (`commands/`)

- **책임**: 명령 오케스트레이션 및 제어 흐름
- **구조**: Command 패턴 구현
  - `init_command.py`: 코퍼스 정제 및 토크나이저 초기화
  - `analyze_command.py`: 토큰 빈도 분석
  - `distill_command.py`: 토크나이저 증류
  - `select_command.py`: IVR 대상 선정
  - `remap_command.py`: 토큰 재할당
  - `align_command.py`: 임베딩 재정렬
  - `train_command.py`: 모델 미세조정
- **역할**:
  - 도메인 로직 조합으로 비즈니스 유스케이스 구현
  - 입출력 경로 관리
  - 실행 결과를 Rich 컴포넌트로 시각화

#### 도메인 계층 (`analysis/`, `tokenizer/`, `embedding/`, `training/`, `corpus/`)

- **책임**: 핵심 비즈니스 로직 및 알고리즘
- **특징**:
  - CLI/Command 계층과 독립적
  - 다른 프로젝트에서 재사용 가능
  - 순수 비즈니스 로직 구현
  - 간결한 로깅만 제공

### 주요 모듈 설명

#### `src/gpt2_ivr/analysis/`

**token_frequency.py**
- 코퍼스 토큰화 및 빈도 집계
- 병렬 처리로 대용량 데이터 효율 처리
- 출력: Parquet 형식 파일

**bpe_corpus_export.py**
- BPE Token ID 시퀀스 생성
- 검증 및 메타데이터 저장

**candidate_selection.py**
- 희생(저빈도) 토큰 식별
- 신규(고빈도 바이그램) 토큰 선정
- 매칭 규칙 생성

#### `src/gpt2_ivr/tokenizer/`

**init_assets.py**
- GPT-2 토크나이저 및 설정 다운로드
- 원본 토크나이저 저장

**distill.py**
- Unigram 토크나이저 학습
- BPE와의 동등성 보장
- 어휘 크기 유지

#### `src/gpt2_ivr/embedding/`

**extract.py**
- GPT-2 모델에서 임베딩 추출
- wte(토큰) 및 wpe(위치) 임베딩 저장

**reorder.py**
- 임베딩 재정렬
- 신규 토큰 임베딩 초기화
- 메타데이터 저장

#### `src/gpt2_ivr/corpus/`

**normalize.py**
- 다양한 형식의 데이터 정규화
- 인코딩 표준화
- 텍스트 정제

#### `src/gpt2_ivr/training/`

**train.py**
- 모델 미세조정 로직 (구현 예정)
- Accelerate 기반 분산 학습

### 경로 상수 관리

모든 artifacts 경로는 `src/gpt2_ivr/constants.py`에서 중앙 관리됩니다.

**주요 상수**:

```python
# 코퍼스 경로
CORPORA_CLEANED_DIR              # artifacts/corpora/cleaned
CORPORA_RAW_DIR                  # artifacts/corpora/raw

# 토크나이저 경로
TOKENIZER_ORIGINAL_DIR           # artifacts/tokenizers/original
TOKENIZER_DISTILLED_UNIGRAM_DIR  # artifacts/tokenizers/distilled_unigram
TOKENIZER_REMAPPED_DIR           # artifacts/tokenizers/remapped

# 분석 산출물 경로
BPE_TOKEN_ID_SEQUENCES_FILE      # artifacts/analysis/reports/bpe_token_id_sequences.txt
TOKEN_FREQUENCY_FILE             # artifacts/analysis/reports/token_frequency.parquet
REPLACEMENT_CANDIDATES_FILE      # artifacts/analysis/reports/replacement_candidates.csv
SELECTION_LOG_FILE               # artifacts/analysis/reports/selection_log.md

# 임베딩 경로
EMBEDDINGS_ROOT                  # artifacts/embeddings

# 학습 경로
TRAINING_CHECKPOINT_DIR          # artifacts/training/sft_checkpoint
```

---

## 📊 데이터 모델 및 구조

### Token Frequency Parquet

**경로**: `artifacts/analysis/reports/token_frequency.parquet`

**스키마**:
| 필드명 | 타입 | 설명 |
|--------|------|------|
| `token_id` | int32 | 토큰 ID (0~50256) |
| `token_str` | string | 토큰 문자열 |
| `frequency` | int64 | 코퍼스 출현 횟수 |

**용도**:
- 저빈도 토큰 식별
- 토큰별 통계 분석

---

### Replacement Candidates CSV

**경로**: `artifacts/analysis/reports/replacement_candidates.csv`

**스키마**:

```
sacrifice_id,sacrifice_token,new_token,sacrifice_freq,new_bigram_freq,confidence
45632,▁<specific>,문제▁해결,12,3450,0.95
45633,▁<rare>,데이터▁분석,8,2890,0.92
...
```

**필드 설명**:
| 필드명 | 설명 |
|--------|------|
| `sacrifice_id` | 희생할 토큰 ID |
| `sacrifice_token` | 희생할 토큰 문자열 |
| `new_token` | 신규 토큰 문자열 (병합된 바이그램) |
| `sacrifice_freq` | 희생 토큰의 빈도 |
| `new_bigram_freq` | 신규 바이그램의 빈도 |
| `confidence` | 교체 신뢰도 (0~1) |

---

### Remap Rules YAML

**경로**: `src/gpt2_ivr/tokenizer/remap_rules.yaml`

**형식**:

```yaml
# token_id: new_token_string
45632: "문제해결"
45633: "데이터분석"
45634: "머신러닝"
...
```

**용도**:
- 토크나이저 어휘 재구성
- 임베딩 재정렬 시 매핑 정보 제공

---

### BPE Token ID Sequences

**경로**: `artifacts/analysis/reports/bpe_token_id_sequences.txt`

**형식**: 한 라인당 한 문서의 token ID 시퀀스

```
50256 12345 678 90 ...
50256 11111 2222 3333 ...
...
```

**용도**:
- Unigram 토크나이저 증류의 라벨 데이터
- 검증 및 분석

---

### 임베딩 메타데이터

**경로**: `artifacts/embeddings/*.json`

**예시 (aligned_wte_metadata.json)**:

```json
{
  "original_vocab_size": 50257,
  "new_vocab_size": 50257,
  "preserved_token_count": 50000,
  "new_tokens": [
    {"id": 45632, "token": "문제해결", "from_sacrifice": 45632},
    ...
  ],
  "init_strategy": "mean",
  "init_source_vocab_size": 50257
}
```

---

## 🔄 데이터 흐름 및 파이프라인

### 전체 파이프라인 흐름

```
1. init (코퍼스 준비)
   └─→ artifacts/corpora/cleaned/*.txt

2. analyze (토큰 분석)
   ├─→ artifacts/analysis/reports/bpe_token_id_sequences.txt
   └─→ artifacts/analysis/reports/token_frequency.parquet

3. distill-tokenizer (토크나이저 증류)
   └─→ artifacts/tokenizers/distilled_unigram/

4. select (IVR 대상 선정)
   ├─→ artifacts/analysis/reports/replacement_candidates.csv
   └─→ artifacts/analysis/reports/selection_log.md

5. remap (토큰 재할당)
   ├─→ src/gpt2_ivr/tokenizer/remap_rules.yaml
   └─→ artifacts/tokenizers/remapped/

6. align (임베딩 재정렬)
   ├─→ artifacts/embeddings/original_wte.pt
   ├─→ artifacts/embeddings/original_wpe.pt
   ├─→ artifacts/embeddings/aligned_wte.pt
   ├─→ artifacts/embeddings/final_wte.pt
   └─→ artifacts/embeddings/*.json

7. train (모델 미세조정)
   └─→ artifacts/training/sft_checkpoint/
```

### 단계별 의존성

| 단계 | 선행 단계 | 입력 | 출력 |
|------|----------|------|------|
| `init` | 없음 | 원본 코퍼스 | 정제된 코퍼스 |
| `analyze` | `init` | 정제 코퍼스, BPE 토크나이저 | Token freq, ID seq |
| `distill` | `analyze` | 코퍼스, BPE seq | Unigram 토크나이저 |
| `select` | `analyze` | Token freq, ID seq | 교체 후보 |
| `remap` | `select`, `distill` | Unigram, 후보 | 재할당 토크나이저 |
| `align` | `remap` | 재할당 토크나이저 | 임베딩 |
| `train` | `align` | 임베딩, 코퍼스 | 미세조정 모델 |

---

## 🛠️ 실행 가이드

### 환경 준비

```bash
# 1. 의존성 동기화
uv sync

# 2. 환경 확인
python --version  # Python 3.13 ~ 3.14
```

### 실행 명령어

#### 1단계: 코퍼스 정제

```bash
uv run ivr init
```

**옵션**:

```bash
uv run ivr init \
  --raw-corpora-dir artifacts/corpora/raw \
  --cleaned-corpora-dir artifacts/corpora/cleaned \
  --text-key text \
  --encoding utf-8 \
  --normalize-force
```

---

#### 2단계: 토큰 분석

```bash
uv run ivr analyze
```

**산출물**:
- `artifacts/analysis/reports/bpe_token_id_sequences.txt`
- `artifacts/analysis/reports/token_frequency.parquet`

---

#### 3단계: 토크나이저 증류

```bash
uv run ivr distill-tokenizer
```

**산출물**:
- `artifacts/tokenizers/distilled_unigram/tokenizer.json`

---

#### 4단계: IVR 대상 선정

```bash
uv run ivr select
```

**산출물**:
- `artifacts/analysis/reports/replacement_candidates.csv`
- `artifacts/analysis/reports/selection_log.md`

---

#### 5단계: 토큰 재할당

```bash
uv run ivr remap
```

**산출물**:
- `src/gpt2_ivr/tokenizer/remap_rules.yaml`
- `artifacts/tokenizers/remapped/tokenizer.json`

---

#### 6단계: 임베딩 재정렬

```bash
uv run ivr align \
  --model-name openai-community/gpt2 \
  --original-tokenizer-dir artifacts/tokenizers/original \
  --remapped-tokenizer-dir artifacts/tokenizers/remapped \
  --remap-rules-path src/gpt2_ivr/tokenizer/remap_rules.yaml \
  --embeddings-output-dir artifacts/embeddings \
  --init-strategy mean
```

**옵션**:
- `--init-strategy`: `mean` (기본), `random`, `zeros`

**산출물**:
- `artifacts/embeddings/original_wte.pt`
- `artifacts/embeddings/aligned_wte.pt`
- `artifacts/embeddings/final_wte.pt`
- `artifacts/embeddings/*.json`

---

#### 7단계: 모델 미세조정

```bash
uv run ivr train
```

**산출물**:
- `artifacts/training/sft_checkpoint/`

---

## 🎨 코딩 스타일 및 컨벤션

### Python 스타일

- **들여쓰기**: 공백 4칸
- **라인 길이**: 120자 (Black 설정)
- **코드 포맷팅**: Black 적용 (`uv run black .`)
- **타입 힌트**: 필수 (내장 제네릭 타입 사용)
- **Docstring**: Google 스타일, 한국어 작성

### 함수/클래스 네이밍

- 함수: `snake_case` (목적이 드러나는 이름)
- 클래스: `PascalCase`
- 상수: `UPPER_SNAKE_CASE`
- Private: `_leading_underscore`

### Docstring 예시

```python
def example_function(param1: str, param2: int) -> bool:
    """함수의 목적을 한 줄로 요약한다.

    더 자세한 설명이 필요한 경우 빈 줄 이후 작성한다.

    Args:
        param1: 첫 번째 파라미터 설명
        param2: 두 번째 파라미터 설명

    Returns:
        반환값 설명

    Raises:
        ValueError: 발생 가능한 예외 설명
    """
```

### 로깅 가이드라인

#### 도메인 레이어 (분석, 토크나이저, 임베딩)

- **목적**: 순수 비즈니스 로직 제공
- **로깅 방식**: 필수 정보만 간결하게
- **어조**: 사실 전달 중심
- **규칙**:
  - 이모지 및 장식 문자 사용 금지
  - "~ 시작", "~ 완료" 형태로 간결하게
  - 들여쓰기 장식(`└─`, `├─` 등) 사용 금지

**예시**:

```python
logger.info("토크나이저 로드: %s", tokenizer_dir)
logger.info("vocab 크기: %d", vocab_size)
logger.info("임베딩 추출 완료")
```

#### 애플리케이션 레이어 (Command)

- **목적**: 사용자 대면 인터페이스
- **로깅**: 디버깅용 최소한의 정보
- **UX**: Rich 라이브러리로 시각적 출력
- **도구**:
  - `rich.table.Table`: 구조화된 결과
  - `rich.panel.Panel`: 단계 구분 및 완료
  - `rich.progress.track`: 진행률 표시

**예시**:

```python
console = Console()
table = Table(title="초기화 완료", show_header=False, title_style="bold green")
table.add_row("vocab 크기", f"{vocab_size:,}")
console.print(table)
```

---

## 📈 비기능 요구사항

### 성능

| 요구사항 | 기준 |
|---------|------|
| **대용량 코퍼스 처리** | 병렬 처리로 GB 규모 텍스트 처리 가능 |
| **메모리 효율성** | 스트리밍 처리로 메모리 사용 최소화 |
| **토크나이저 동등성** | BPE와 100% 동일한 encode/decode 결과 |

### 신뢰성

| 요구사항 | 기준 |
|---------|------|
| **재현성** | 동일 입력에 대해 항상 동일한 결과 |
| **데이터 무결성** | 모든 중간 산출물 검증 및 저장 |
| **정적 분석** | Pyright 타입 체크 0 에러 |

### 확장성

| 요구사항 | 기준 |
|---------|------|
| **도메인 커스터마이징** | Remap 규칙으로 쉬운 조정 |
| **코퍼스 확장** | 새 데이터 추가로 파이프라인 재실행 가능 |
| **모듈 재사용** | 도메인 레이어 독립적 재사용 가능 |

### 보안 & 유지보수

| 요구사항 | 기준 |
|---------|------|
| **비밀값 관리** | `.env` 파일로 민감 정보 분리 |
| **코드 품질** | Black 포맷팅, Pyright 타입 체크 |
| **마크다운 검증** | `npx markdownlint-cli` 사용 |

---

## 🚀 향후 개선 방향

### 단기 (1-2개월)

1. **Training 모듈 구현**
   - Accelerate 기반 분산 학습
   - 학습 메트릭 및 로깅
   - 체크포인트 저장 및 복구

2. **검증 및 테스트**
   - 자동화된 단위 테스트 작성
   - 파이프라인 통합 테스트
   - 엔드투엔드 테스트

3. **문서화**
   - API 문서 자동생성 (Sphinx)
   - 튜토리얼 및 예제 추가
   - 문제 해결 가이드

### 중기 (2-4개월)

1. **성능 최적화**
   - 코퍼스 처리 속도 개선
   - 메모리 사용 최소화
   - 병렬 처리 확대

2. **도메인 확장**
   - 다양한 도메인별 코퍼스 지원
   - 커스텀 토크나이저 모델 지원
   - 다국어 지원

3. **모니터링 & 분석**
   - 실시간 진행률 대시보드
   - 분석 리포트 자동화
   - 성능 메트릭 수집

### 장기 (4개월 이상)

1. **멀티 토크나이저 지원**
   - WordPiece, SentencePiece 등 다양한 토크나이저 지원
   - 토크나이저 간 마이그레이션 자동화

2. **모델 아키텍처 확장**
   - GPT-2 외 다양한 기본 모델 지원
   - BERT, T5 등 다른 모델 구조 적응

3. **연구 도구화**
   - 웹 대시보드 제공
   - REST API 서버 제공
   - 클라우드 통합 (HuggingFace Hub, etc.)

---

## 📝 산출물 요약

### 핵심 연구 자산

프로젝트의 가장 중요한 결과물은 **코드보다 생성된 분석 자료**입니다.

| 산출물 | 경로 | 용도 |
|--------|------|------|
| **토큰 빈도** | `artifacts/analysis/reports/token_frequency.parquet` | 토큰 통계 분석 |
| **BPE 시퀀스** | `artifacts/analysis/reports/bpe_token_id_sequences.txt` | 증류 라벨 데이터 |
| **교체 후보** | `artifacts/analysis/reports/replacement_candidates.csv` | IVR 규칙 생성 |
| **선정 로그** | `artifacts/analysis/reports/selection_log.md` | 프로세스 추적 |
| **원본 토크나이저** | `artifacts/tokenizers/original/` | 기준점 |
| **증류 토크나이저** | `artifacts/tokenizers/distilled_unigram/` | 마이그레이션 결과 |
| **재할당 토크나이저** | `artifacts/tokenizers/remapped/` | IVR 결과 |
| **임베딩** | `artifacts/embeddings/` | 모델 미세조정 기초 |
| **학습 체크포인트** | `artifacts/training/sft_checkpoint/` | 최종 모델 |

---

## 🔗 참고 자료

### 프로젝트 구조

```
gpt2-ivr-claude/
├── src/gpt2_ivr/
│   ├── cli.py                    # CLI 진입점
│   ├── constants.py              # 경로 상수 중앙 관리
│   ├── commands/                 # Command 패턴 구현
│   │   ├── base.py              # Command 추상 클래스
│   │   ├── init_command.py
│   │   ├── analyze_command.py
│   │   ├── distill_command.py
│   │   ├── select_command.py
│   │   ├── remap_command.py
│   │   ├── align_command.py
│   │   └── train_command.py
│   ├── analysis/                 # 분석 로직
│   │   ├── token_frequency.py
│   │   ├── candidate_selection.py
│   │   └── bpe_corpus_export.py
│   ├── tokenizer/                # 토크나이저 로직
│   │   ├── init_assets.py
│   │   ├── distill.py
│   │   └── remap_rules.yaml
│   ├── embedding/                # 임베딩 처리
│   │   ├── extract.py
│   │   ├── reorder.py
│   │   └── init_new.py
│   ├── training/                 # 학습 로직
│   │   └── train.py
│   ├── corpus/                   # 코퍼스 처리
│   │   └── normalize.py
│   └── utils/                    # 유틸리티
├── artifacts/                    # 산출물 저장
│   ├── corpora/
│   │   ├── raw/
│   │   └── cleaned/
│   ├── analysis/reports/
│   ├── tokenizers/
│   ├── embeddings/
│   ├── training/
│   └── logs/
├── scripts/                      # 보조 스크립트
├── tests/                        # 테스트 코드 (계획)
├── README.md
├── AGENTS.md
├── pyproject.toml
└── uv.lock
```

### 주요 문서

- **README.md**: 프로젝트 개요 및 빠른 시작
- **AGENTS.md**: 개발 가이드라인 및 아키텍처
- **PRD.md** (본 문서): 제품 요구사항 명세
- **CLAUDE.md**: 개발 에이전트 인스트럭션

---

## 📞 질문 및 피드백

이 문서는 지속적으로 업데이트됩니다.

- 버그 리포트: GitHub Issues
- 기능 요청: GitHub Discussions
- 문서 개선: Pull Requests

---

**최종 업데이트**: 2026-02-10
**문서 버전**: 1.0
**작성자**: Claude Code (Anthropic)
