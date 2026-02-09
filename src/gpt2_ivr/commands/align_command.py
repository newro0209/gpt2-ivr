"""Embedding Alignment Command"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gpt2_ivr.commands.base import Command
from gpt2_ivr.embedding import (
    extract_embeddings,
    initialize_new_token_embeddings,
    reorder_embeddings,
)
from gpt2_ivr.utils.logging_config import get_logger


class AlignCommand(Command):
    """Align Command - 임베딩 추출, 재정렬 및 초기화를 수행한다."""

    def __init__(
        self,
        model_name: str = "openai-community/gpt2",
        original_tokenizer_dir: Path = Path("artifacts/tokenizers/original"),
        remapped_tokenizer_dir: Path = Path("artifacts/tokenizers/remapped"),
        remap_rules_path: Path = Path("src/gpt2_ivr/tokenizer/remap_rules.yaml"),
        embeddings_output_dir: Path = Path("artifacts/embeddings"),
        init_strategy: str = "mean",
    ) -> None:
        self.logger = get_logger("gpt2_ivr.align")
        self.model_name = model_name
        self.original_tokenizer_dir = original_tokenizer_dir
        self.remapped_tokenizer_dir = remapped_tokenizer_dir
        self.remap_rules_path = remap_rules_path
        self.embeddings_output_dir = embeddings_output_dir
        self.init_strategy = init_strategy

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """커맨드 실행 로직"""
        self.logger.info("🚀 Align Command 시작")

        # 1. 원본 모델에서 임베딩 추출
        self.logger.info("=" * 60)
        self.logger.info("1단계: 원본 모델 임베딩 추출")
        self.logger.info("=" * 60)

        extract_result = extract_embeddings(
            model_name=self.model_name,
            output_dir=self.embeddings_output_dir,
            logger=self.logger,
        )

        # 2. Remap 규칙에 따라 임베딩 재정렬
        self.logger.info("=" * 60)
        self.logger.info("2단계: 임베딩 재정렬")
        self.logger.info("=" * 60)

        reorder_result = reorder_embeddings(
            original_wte_path=extract_result["wte"],
            original_tokenizer_dir=self.original_tokenizer_dir,
            remapped_tokenizer_dir=self.remapped_tokenizer_dir,
            remap_rules_path=self.remap_rules_path,
            output_dir=self.embeddings_output_dir,
            logger=self.logger,
        )

        # 3. 신규 토큰 임베딩 초기화
        self.logger.info("=" * 60)
        self.logger.info("3단계: 신규 토큰 임베딩 초기화")
        self.logger.info("=" * 60)

        init_result = initialize_new_token_embeddings(
            aligned_wte_path=reorder_result["aligned_wte"],
            original_tokenizer_dir=self.original_tokenizer_dir,
            remapped_tokenizer_dir=self.remapped_tokenizer_dir,
            remap_rules_path=self.remap_rules_path,
            output_dir=self.embeddings_output_dir,
            init_strategy=self.init_strategy,
            logger=self.logger,
        )

        self.logger.info("=" * 60)
        self.logger.info("✅ Align Command 완료")
        self.logger.info("=" * 60)

        return {
            "status": "success",
            "extract_result": extract_result,
            "reorder_result": reorder_result,
            "init_result": init_result,
            "embeddings_dir": str(self.embeddings_output_dir),
        }

    def get_name(self) -> str:
        """커맨드 이름 반환"""
        return "align"
