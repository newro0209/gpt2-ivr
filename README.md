# gpt2-ivr

> **BPE → Unigram Tokenizer Distillation 이후 IVR를 수행하는 연구/실험 표준 구조**

이 프로젝트는 단순 파인튜닝 프로젝트가 아닙니다.
목표는 **사전학습된 GPT-2의 토크나이저 모델을 안전하게 교체**한 뒤,
그 위에서 **Vocabulary Reassignment(IVR)** 를 수행하는 재현 가능한 연구 파이프라인을 구축하는 것입니다.

## 프로젝트 목표

아래 단계를 **하나의 파이프라인으로 연속 수행**합니다.

1. GPT-2의 **BPE 토크나이저를 Unigram 토크나이저로 교체**
2. Unigram 토크나이저가 **BPE와 완전히 동일한**
   - 입력 → token id 시퀀스
   - token id 시퀀스 → 디코딩 결과

   를 만들도록 학습 (**Tokenizer Distillation**)
3. 그 위에서 **IVR(In-place Vocabulary Reassignment)** 수행
4. embedding 재정렬 후 미세조정

즉, 이 프로젝트의 핵심은 다음과 같습니다.

> **Tokenizer Model Migration + IVR**

## 실행 파이프라인 (엔트리 포인트)

모든 단계는 엔트리 포인트로만 실행합니다.

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

## Tokenizer Distillation (핵심 개념)

이 단계의 목적은 **토큰이나 id를 바꾸는 것이 아닙니다.**

> **토크나이저 "모델"만 BPE → Unigram으로 교체**하되
> 모델이 받는 token id 시퀀스를 완전히 동일하게 유지하는 것

### Distillation 이후 만족해야 하는 조건

| 항목 | 상태 |
| --- | --- |
| id ↔ token | GPT-2와 동일 |
| encode 결과 | GPT-2와 동일 |
| decode 결과 | GPT-2와 동일 |
| tokenizer 모델 | Unigram (merges 없음) |

### 방법

1. GPT-2 BPE로 코퍼스를 전부 토큰화하여 **정답 token id 시퀀스** 생성
2. 이 시퀀스를 Unigram 학습의 라벨로 사용
3. vocab size를 50257로 유지

결과적으로 모델은 **토크나이저가 바뀐 것을 인지하지 못합니다.**

## IVR 단계

Distilled Unigram 토크나이저 위에서 저빈도 토큰을 도메인 고빈도 토큰으로 교체합니다.

```text
replacement_candidates.csv
        ↓
remap_rules.yaml
        ↓
src/gpt2_ivr/embedding/reorder.py
        ↓
src/gpt2_ivr/training/train.py
```

Distillation은 "안 깨지게 옮기는 단계",
IVR은 "토큰 표현력을 개선하는 단계"입니다.

## 산출물 (연구 자산)

### 분석 리포트

```text
artifacts/analysis/reports/
├─ token_frequency.parquet
├─ replacement_candidates.csv
├─ bpe_token_id_sequences.txt
└─ selection_log.md
```

### 토크나이저

- `artifacts/tokenizers/original/`
- `artifacts/tokenizers/distilled_unigram/`
- `artifacts/tokenizers/remapped/`

### 임베딩

- `artifacts/embeddings/original_wte.pt` - 원본 토큰 임베딩
- `artifacts/embeddings/original_wpe.pt` - 원본 위치 임베딩
- `artifacts/embeddings/aligned_wte.pt` - 재정렬된 토큰 임베딩
- `artifacts/embeddings/final_wte.pt` - 최종 임베딩 (초기화 완료)
- `artifacts/embeddings/*.json` - 각 단계별 메타데이터

### 학습

- `artifacts/training/` - 학습 체크포인트 및 로그

이 파일들은 코드보다 더 중요한 **연구 기록**입니다.

## 환경 및 도구

| 항목 | 스택 |
| --- | --- |
| 환경 관리 | uv |
| Python | 3.13 ~ 3.14 |
| Tokenizer | Hugging Face `tokenizers` (Unigram) |
| Training | Hugging Face `accelerate` |
| Base Model | `openai-community/gpt2` |
| CUDA | 13.0 |
| PyTorch | 2.10 |

## 빌드, 테스트, 개발 명령어

- 환경 동기화는 `uv sync`를 사용한다.
- 전체 파이프라인은 아래 순서로 실행한다.
  - `uv run ivr analyze`
  - `uv run ivr distill-tokenizer`
  - `uv run ivr select`
  - `uv run ivr remap`
  - `uv run ivr align`
  - `uv run ivr train`
- `uv run ivr distill-tokenizer` 단계는 `uv run ivr remap` 이전에 반드시 수행한다.
- 엔트리 포인트 실행은 개별 스크립트 직접 호출 대신 `uv run ivr <command>` 형식을 기본으로 사용한다.
- `uv run ivr init`은 `artifacts/corpora/raw/` 아래의 `.txt`, `.jsonl`, `.json`을 일관된 `.txt`로 정제하여 `artifacts/corpora/cleaned/`에 저장한다. `--text-key`, `--encoding`, `--raw-corpora-dir`, `--cleaned-corpora-dir`, `--normalize-force`로 정제 동작을 미세 조정할 수 있다.

## Quick Start

### 1. 환경 준비

```bash
uv sync
```

> Python 3.13~3.14, CUDA 13.0, PyTorch 2.10 환경을 전제로 합니다.

### 2. 코퍼스 준비

```text
artifacts/corpora/raw/     # 원본 데이터 수집
artifacts/corpora/cleaned/ # 정제 완료 데이터
```

> `uv run ivr init`을 실행하면 `artifacts/corpora/raw/` 아래의 `.txt`, `.jsonl`, `.json` 파일을
> 자동으로 정제하여 `artifacts/corpora/cleaned/`에 일관된 `.txt` 형식으로 저장합니다.
> 기본적으로 JSON/JSONL에서 `text` 키를 사용하며 인코딩은 `utf-8`입니다.
> 필요하면 `--text-key`, `--encoding`, `--normalize-force`, `--raw-corpora-dir`,
> `--cleaned-corpora-dir`로 동작을 조정할 수 있습니다.

### 3. 모델 및 토크나이저 초기화

```bash
uv run ivr init
```

- Hugging Face Hub에서 GPT-2 토크나이저와 모델 설정을 다운로드
- `--force` 옵션으로 기존 파일이 있어도 다시 다운로드 가능
- `--raw-corpora-dir`/`--cleaned-corpora-dir`를 지정하여 다른 디렉토리를 정제 대상으로 사용할 수 있습니다.
- `--text-key`/`--encoding`으로 JSON 계열 파일에서 읽을 텍스트 키와 인코딩을 조정하거나 `--normalize-force`를 사용하여 존재하는 정제본을 덮어쓸 수 있습니다.

### 4. BPE 토큰 시퀀스 생성

```bash
uv run ivr analyze
```

- GPT-2 BPE 기준 token id 시퀀스를 생성

### 5. Tokenizer Distillation (BPE → Unigram)

```bash
uv run ivr distill-tokenizer
```

- BPE와 동일한 encode/decode를 만드는 Unigram tokenizer 생성

### 6. IVR 대상 토큰 선정

```bash
uv run ivr select
```

- 저빈도 토큰 분석

### 7. 토큰 교체 및 tokenizer 생성

```bash
uv run ivr remap
```

- IVR 적용 tokenizer 생성

### 8. Embedding 재정렬

```bash
uv run ivr align
```

- GPT-2 모델에서 토큰 임베딩(wte)과 위치 임베딩(wpe) 추출
- Remap 규칙에 따라 임베딩 재정렬
- 신규 추가된 토큰에 대한 임베딩 초기화

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

### 9. 미세조정

```bash
uv run ivr train
```

- accelerate 기반 학습 수행
