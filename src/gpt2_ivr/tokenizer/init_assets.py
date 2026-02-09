"""모델 및 토크나이저 초기화 (다운로드) 모듈"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from transformers import AutoTokenizer, AutoConfig

from gpt2_ivr.utils.logging_config import get_logger

logger = get_logger(__name__)


class InitResult(TypedDict):
    """초기화 결과 타입"""

    tokenizer_dir: Path
    vocab_size: int
    model_name: str


def initialize_assets(
    model_name: str,
    tokenizer_dir: Path,
    force: bool,
) -> InitResult:
    """Hugging Face Hub에서 GPT-2 토크나이저와 모델 설정을 다운로드한다.

    Args:
        model_name: Hugging Face Hub 모델 이름
        tokenizer_dir: 토크나이저 저장 디렉토리
        force: True이면 기존 파일이 있어도 다시 다운로드

    Returns:
        초기화 결과 정보를 담은 딕셔너리
    """
    logger.info("🚀 모델 초기화를 시작합니다: %s", model_name)

    # 이미 토크나이저가 존재하는지 확인
    tokenizer_files = list(tokenizer_dir.glob("*")) if tokenizer_dir.exists() else []
    has_tokenizer = any(
        f.name in ["tokenizer.json", "vocab.json", "merges.txt"]
        for f in tokenizer_files
    )

    if has_tokenizer and not force:
        logger.info("✅ 토크나이저가 이미 존재합니다: %s", tokenizer_dir)
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        vocab_size = len(tokenizer.get_vocab())
        logger.info("  └─ vocab 크기: %d", vocab_size)
        return InitResult(
            tokenizer_dir=tokenizer_dir,
            vocab_size=vocab_size,
            model_name=model_name,
        )

    # Hub에서 토크나이저 다운로드
    logger.info("📥 Hugging Face Hub에서 토크나이저를 다운로드합니다: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(tokenizer_dir))

    vocab_size = len(tokenizer.get_vocab())
    logger.info("✅ 토크나이저 다운로드 완료: %s", tokenizer_dir)
    logger.info("  └─ vocab 크기: %d", vocab_size)

    # 모델 설정 다운로드 (가중치는 제외, 설정만 저장)
    logger.info("📥 모델 설정(config)을 다운로드합니다: %s", model_name)
    config = AutoConfig.from_pretrained(model_name)
    config.save_pretrained(str(tokenizer_dir))
    logger.info("✅ 모델 설정 저장 완료: %s", tokenizer_dir)

    logger.info("🎉 초기화 완료.")
    return InitResult(
        tokenizer_dir=tokenizer_dir,
        vocab_size=vocab_size,
        model_name=model_name,
    )
