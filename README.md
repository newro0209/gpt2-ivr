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

   * 입력 → token id 시퀀스
   * token id 시퀀스 → 디코딩 결과
     를 만들도록 학습 (**Tokenizer Distillation**)
3. 그 위에서 **IVR(In‑place Vocabulary Reassignment)** 수행
4. embedding 재정렬 후 미세조정

즉, 이 프로젝트의 핵심은:

> **Tokenizer Model Migration + IVR**

입니다.

---

## 🗂️ 디렉토리 구조

``` 
gpt2-ivr/
├─ README.md                    # 프로젝트 개요, 파이프라인, 실행 방법 문서
├─ pyproject.toml               # 패키지 메타데이터, 의존성, 엔트리 포인트 설정
├─ uv.lock                      # uv 의존성 락파일(재현 가능한 환경 고정)
│
├─ scripts/                     # 파이프라인 외 보조 유틸 스크립트
│   ├─ set_internal_pypi_index.*    # 내부 PyPI 인덱스 설정
│   └─ unset_internal_pypi_index.*  # 내부 PyPI 인덱스 해제
│
├─ corpora/                     # 코퍼스 데이터 저장 루트
│   ├─ raw/                     # 원본 수집 데이터
│   └─ cleaned/                 # 전처리/정제 완료 데이터
│
├─ analysis/                    # 분석 및 후보 선정 로직
│   ├─ token_frequency.py       # 토큰 빈도 통계 계산
│   ├─ candidate_selection.py   # IVR 교체 후보 토큰 선정
│   ├─ bpe_corpus_export.py     # GPT-2 BPE 기준 토큰 시퀀스 추출
│   └─ reports/                 # 분석 결과 산출물 저장
│
├─ tokenizer/                   # 토크나이저 자산 및 규칙
│   ├─ original/                # 원본 GPT-2 토크나이저 보관
│   ├─ distilled_unigram/       # Distillation 완료 Unigram 토크나이저
│   ├─ remapped/                # IVR 적용 후 토크나이저
│   └─ remap_rules.yaml         # 토큰 재할당 규칙 정의
│
├─ embedding/                   # 임베딩 추출/재배치/초기화 로직
│   ├─ extract.py               # 기존 모델 임베딩 추출
│   ├─ reorder.py               # remap 규칙 기준 임베딩 재정렬
│   └─ init_new.py              # 신규 토큰 임베딩 초기화
│
├─ training/                    # 학습 설정 및 학습 실행 코드
│   ├─ sft_config.yaml          # 미세조정 하이퍼파라미터/런타임 설정
│   └─ train.py                 # accelerate 기반 학습 실행
│
└─ src/                         # 패키지 소스 루트
    └─ ivr/                     # 파이프라인 오케스트레이션 패키지
        ├─ cli.py               # `uv run ivr ...` CLI 엔트리 포인트
        ├─ analyze.py           # analyze 단계 오케스트레이션
        ├─ distill_tokenizer.py # distill-tokenizer 단계 오케스트레이션
        ├─ select.py            # select 단계 오케스트레이션
        ├─ remap.py             # remap 단계 오케스트레이션
        ├─ align.py             # align 단계 오케스트레이션
        └─ train.py             # train 단계 오케스트레이션
```

---

## ▶️ 실행 파이프라인 (엔트리 포인트)

모든 단계는 엔트리 포인트를 통해 실행합니다.

```
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

| 항목           | 상태                  |
| ------------ | ------------------- |
| id ↔ token   | GPT‑2와 동일           |
| encode 결과    | GPT‑2와 동일           |
| decode 결과    | GPT‑2와 동일           |
| tokenizer 모델 | Unigram (merges 없음) |

### 방법

1. GPT‑2 BPE로 코퍼스를 전부 토큰화하여 **정답 token id 시퀀스** 생성
2. 이 시퀀스를 Unigram 학습의 label로 사용
3. vocab size = 50257 유지

결과적으로 모델은 **토크나이저가 바뀐 것을 인지하지 못합니다.**

---

## 🔧 IVR 단계

Distilled Unigram 위에서 저빈도 토큰을 도메인 고빈도 토큰으로 교체합니다.

```
replacement_candidates.csv
        ↓
remap_rules.yaml
        ↓
embedding/reorder.py
        ↓
train.py
```

Distillation은 “안 깨지게 옮기는 단계”,
IVR은 “토큰 표현력을 개선하는 단계”입니다.

---

## 📁 분석 산출물 (연구 자산)

```
analysis/reports/
├─ token_frequency.parquet
├─ replacement_candidates.csv
├─ bpe_token_id_sequences.txt
└─ selection_log.md
```

이 파일들은 코드보다 더 중요한 **연구 기록**입니다.

---

## 🧩 역할 분리 원칙

| 위치            | 역할                       |
| ------------- | ------------------------ |
| `src/ivr/*`   | 파이프라인 제어 (Orchestration) |
| `analysis/*`  | 분석 로직 (Research Library) |
| `tokenizer/*` | 토크나이저 산출물                |

---

## 🧰 환경 및 도구

| 항목         | 스택                                  |
| ---------- | ----------------------------------- |
| 환경 관리      | uv                                  |
| Python     | 3.13 ~ 3.14                         |
| Tokenizer  | Hugging Face `tokenizers` (Unigram) |
| Training   | Hugging Face `accelerate`           |
| Base Model | `openai-community/gpt2`             |
| CUDA       | 13.0                                |
| PyTorch    | 2.10                                |

---

## ✅ 이 구조가 보장하는 것

* BPE → Unigram 안전 이식
* 그 위에서 IVR 수행
* 분석 결과의 파일 기반 축적
* 재현 가능한 엔드투엔드 파이프라인

---

## 🚀 Quick Start

### 1️⃣ 환경 준비

```bash
uv sync
```

> Python 3.13~3.14, CUDA 13.0, PyTorch 2.10 환경을 전제로 합니다.

---

### 2️⃣ 코퍼스 준비

```
corpora/raw/     # 원본 데이터 수집
corpora/cleaned/ # 정제 완료 데이터
```

---

### 3️⃣ BPE 토큰 시퀀스 생성

```bash
uv run ivr analyze
```

* GPT‑2 BPE 기준 token id 시퀀스를 생성
* `analysis/reports/bpe_token_id_sequences.txt` 생성

---

### 4️⃣ Tokenizer Distillation (BPE → Unigram)

```bash
uv run ivr distill-tokenizer
```

* BPE와 동일한 encode/decode를 만드는 Unigram tokenizer 생성
* 결과: `tokenizer/distilled_unigram/`

---

### 5️⃣ IVR 대상 토큰 선정

```bash
uv run ivr select
```

* 저빈도 토큰 분석
* `replacement_candidates.csv` 생성

---

### 6️⃣ 토큰 교체 및 tokenizer 생성

```bash
uv run ivr remap
```

* IVR 적용 tokenizer 생성
* 결과: `tokenizer/remapped/`

---

### 7️⃣ Embedding 재정렬

```bash
uv run ivr align
```

* GPT‑2 embedding을 새 tokenizer id 순서에 맞게 재배치

---

### 8️⃣ 미세조정

```bash
uv run ivr train
```

* accelerate 기반 학습 수행
