# 저장소 가이드라인

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

## 역할 분리 원칙

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

## 아키텍처 계층 구조

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

### 횡단 관심사 처리

로깅, 진행률 표시, 배너 같은 공통 기능은 `utils/`를 만들지 않고 `src/gpt2_ivr/cli.py`에서만 준비합니다. CLI가 Rich 콘솔 핸들러, ASCII 배너, 테이블/패널, 로그 파일 흐름을 직접 설정하고, `commands/`나 `analysis/`가 필요할 때 가져다 씁니다. 이 방식으로 별도의 계층 없이 공통 도구를 공유합니다.

### 계층 간 의존성 규칙

```text
프레젠테이션 계층 (cli.py)
        ↓
애플리케이션 계층 (commands/)
        ↓
도메인 계층 (analysis/, tokenizer/, embedding/, training/)
```

- **단방향 의존성**: 상위 계층은 하위 계층에만 의존
- **도메인 독립성**: 도메인 계층은 CLI/Command 계층을 알지 못함
- **재사용성**: 각 계층은 독립적으로 테스트 및 재사용 가능

## 중앙화된 경로 상수 관리

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

## 코딩 스타일 및 네이밍 규칙

- Python 들여쓰기는 공백 4칸을 사용한다.
- 함수명은 `snake_case`를 사용하고 목적이 드러나는 이름을 사용한다.
- CLI 오케스트레이션 로직은 `src/gpt2_ivr/cli.py`와 `src/gpt2_ivr/commands/`로 분리한다.
- 분석 코드와 실행 제어 코드를 혼합하지 않는다.
- 토크나이저/임베딩/학습 단계 간 입출력 경로를 명시적으로 유지한다.
- PEP 규칙을 엄격히 준수한다(PEP 8, PEP 257, PEP 484, PEP 526, PEP 544 포함).
- 코드 포매팅은 Black Formatter를 사용한다.
- 포매팅 검증 또는 적용은 `uv run black .` 명령을 기준으로 수행한다.
- Black 설정: `line-length = 120`, `target-version = ["py313"]` (pyproject.toml 참조)
- 타입 힌트 작성 시 `list`, `dict`, `str` 등 내장 제네릭 타입을 우선 사용한다.
- Python 코드는 항상 Pythonic한 방식으로 작성한다.
- 짧더라도 복잡한 로직에는 의도를 설명하는 주석을 추가한다.
- 복잡한 흐름 주석은 `1) ... 2) ...`처럼 단계 번호를 붙이고 불필요한 주석은 제거한다.
- 병렬 처리 가능 구간에서는 `concurrent.futures` 활용을 우선 검토한다.
- 진행 상태 추적을 위해 로깅을 기본 적용한다.
- 시간이 오래 걸리는 작업에는 프로그래스바를 적용해 실행 가시성을 확보한다.
- 주석, 문서, 로그 메시지는 한국어로 작성한다.
- 가독성 개선 목적의 이모지 사용을 허용한다.

### 로깅 및 UX 가이드라인

프로젝트는 **도메인 레이어**(비즈니스 로직)와 **어플리케이션 레이어**(CLI 커맨드)로 계층화되어 있으며, 각 레이어는 다른 접근 방식을 따른다.

#### 도메인 레이어 (analysis/, embedding/, tokenizer/)

- **목적**: 재사용 가능한 순수 비즈니스 로직 제공
- **로깅**: 필수 정보만 간결하게 로깅
- **UX**: 신경 쓰지 않음 (어플리케이션 레이어가 담당)
- **어조**: 통일되고 간결한 사실 전달
  - ✅ "토크나이저 로드: %s"
  - ✅ "임베딩 추출 완료"
  - ❌ "🚀 토크나이저를 로드합니다..."
  - ❌ "✅ 임베딩 추출이 완료되었습니다!"
- **규칙**:
  - 이모지 및 장식 문자 사용 금지
  - "~를 시작합니다", "~했습니다" 등 장황한 표현 지양
  - "~ 시작", "~ 완료" 형태로 간결하게 작성
  - 들여쓰기 장식(`└─`, `├─` 등) 사용 금지
  - 중요한 단계 전환 또는 에러/경고만 로깅

#### 어플리케이션 레이어 (commands/)

- **목적**: 사용자 대면 인터페이스 제공
- **로깅**: 디버깅용 최소한의 로깅만 유지
- **UX**: Rich 라이브러리로 시각적 출력 제공
- **도구**:
  - `rich.table.Table`: 구조화된 결과 출력
  - `rich.panel.Panel`: 단계 구분 및 완료 메시지
  - `rich.progress.track`: 진행 상황 표시
  - `rich.console.Console`: 통합 출력 관리
- **규칙**:
  - 사용자 대면 출력은 Rich 컴포넌트 사용
  - logger.info는 디버깅 정보만 남김 (사용자는 보지 않음)
  - 색상, 테두리, 정렬로 가독성 향상
  - 결과 요약은 Table 또는 Panel로 표시

#### 로깅 메시지 작성 예시

```python
# 도메인 레이어 - 간결한 로깅
logger.info("토크나이저 로드: %s", tokenizer_dir)
logger.info("vocab 크기: %d", vocab_size)
logger.info("임베딩 추출 완료")

# 어플리케이션 레이어 - Rich 기반 UX
console = Console()
table = Table(title="초기화 완료", show_header=False, title_style="bold green")
table.add_row("vocab 크기", f"{vocab_size:,}")
table.add_row("토크나이저 경로", str(tokenizer_dir))
console.print(table)

panel = Panel(
    "[bold cyan]단계 완료[/bold cyan]\n\n경로: /path/to/output",
    title="결과",
    border_style="green"
)
console.print(panel)
```

### Docstring 작성 규칙

- **모든 모듈, 클래스, 공개 함수/메서드에 docstring을 작성한다.**
- **Google 스타일 docstring**을 사용한다.
- Docstring은 **한국어**로 작성한다.
- 구조:

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

      Yields:
          제너레이터의 경우 yield 값 설명
      """
  ```

- 한 줄 요약은 마침표로 끝낸다.
- 타입 정보는 타입 힌트에 이미 있으므로 docstring에 중복 작성하지 않는다.
- TypedDict와 dataclass에는 Attributes 섹션을 추가한다.

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

## Git 브랜치 전략 및 워크플로우

### 브랜치 생성 규칙

- **모든 새로운 작업은 반드시 별도 브랜치를 생성하고 시작한다.**
- master 브랜치에서 직접 작업하지 않는다.
- 브랜치 이름은 다음 컨벤션을 따른다:
  - 기능 추가: `feature/<기능명>` (예: `feature/add-validation`)
  - 리팩토링: `refactor/<대상>` (예: `refactor/cli-simplification`)
  - 버그 수정: `fix/<버그명>` (예: `fix/tokenizer-crash`)
  - 문서 작업: `docs/<문서명>` (예: `docs/update-readme`)
  - 실험/테스트: `experiment/<실험명>` (예: `experiment/new-algorithm`)

### 작업 흐름

1. **작업 시작**: 새 브랜치 생성

   ```bash
   git checkout -b <브랜치-타입>/<작업명>
   ```

2. **작업 진행**: 브랜치에서 코드 작성 및 커밋
   - 커밋 메시지는 Gitmoji 포함, 한국어 작성
   - 한 커밋에는 한 단계의 논리적 변경만 포함

3. **작업 완료**: 브랜치 푸시 및 master 머지

   ```bash
   git push -u origin <브랜치명>
   git checkout master
   git merge <브랜치명> --no-ff
   git push origin master
   ```

### 머지 전략

- `--no-ff` 플래그를 사용하여 머지 커밋을 명시적으로 생성한다.
- Fast-forward 머지를 피하여 브랜치 히스토리를 명확히 유지한다.
- 머지 커밋 메시지에도 Gitmoji와 간결한 요약을 포함한다.

## 커밋 및 Pull Request 가이드라인

- 커밋 메시지는 Gitmoji를 포함해 작성한다.
- 커밋 제목과 본문은 한국어로 작성한다.
- 한 커밋에는 한 단계의 논리적 변경만 포함한다.
- PR 제목은 Gitmoji를 포함하고 한국어로 작성한다.
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
