"""중앙화된 산출물 경로 상수 관리

이 모듈은 프로젝트 전체에서 사용되는 artifacts 경로를 중앙에서 관리한다.
모든 하드코딩된 경로는 이 모듈의 상수를 참조해야 한다.
"""

from __future__ import annotations

from pathlib import Path

# ====================================================================
# 📁 루트 디렉토리
# ====================================================================

ARTIFACTS_ROOT = Path("artifacts")

# ====================================================================
# 📊 코퍼스 경로
# ====================================================================

CORPORA_ROOT = ARTIFACTS_ROOT / "corpora"
CORPORA_RAW_DIR = CORPORA_ROOT / "raw"
CORPORA_CLEANED_DIR = CORPORA_ROOT / "cleaned"

# ====================================================================
# 🔤 토크나이저 경로
# ====================================================================

TOKENIZERS_ROOT = ARTIFACTS_ROOT / "tokenizers"
TOKENIZER_ORIGINAL_DIR = TOKENIZERS_ROOT / "original"
TOKENIZER_DISTILLED_UNIGRAM_DIR = TOKENIZERS_ROOT / "distilled_unigram"
TOKENIZER_REMAPPED_DIR = TOKENIZERS_ROOT / "remapped"

# ====================================================================
# 📈 분석 산출물 경로
# ====================================================================

ANALYSIS_ROOT = ARTIFACTS_ROOT / "analysis"
ANALYSIS_REPORTS_DIR = ANALYSIS_ROOT / "reports"

# 분석 리포트 파일
BPE_TOKEN_ID_SEQUENCES_FILE = ANALYSIS_REPORTS_DIR / "bpe_token_id_sequences.txt"
TOKEN_FREQUENCY_FILE = ANALYSIS_REPORTS_DIR / "token_frequency.parquet"
REPLACEMENT_CANDIDATES_FILE = ANALYSIS_REPORTS_DIR / "replacement_candidates.csv"
SELECTION_LOG_FILE = ANALYSIS_REPORTS_DIR / "selection_log.md"

# ====================================================================
# 🧮 임베딩 경로
# ====================================================================

EMBEDDINGS_ROOT = ARTIFACTS_ROOT / "embeddings"

# ====================================================================
# 📝 로그 경로
# ====================================================================

LOGS_DIR = ARTIFACTS_ROOT / "logs"

# ====================================================================
# 🏋️ 학습 경로
# ====================================================================

TRAINING_ROOT = ARTIFACTS_ROOT / "training"
TRAINING_CHECKPOINT_DIR = TRAINING_ROOT / "sft_checkpoint"

# ====================================================================
# ⚙️ 기타 설정 경로
# ====================================================================

REMAP_RULES_PATH = Path("src/gpt2_ivr/tokenizer/remap_rules.yaml")
