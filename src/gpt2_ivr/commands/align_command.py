"""Embedding Alignment Command"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gpt2_ivr.commands.base import Command
from gpt2_ivr.embedding.extract import extract_embeddings
from gpt2_ivr.embedding.init_new import initialize_new_embeddings
from gpt2_ivr.embedding.reorder import reorder_embeddings
from gpt2_ivr.utils.logging_config import get_logger


class AlignCommand(Command):
    """Align Command - 임베딩 추출, 재정렬, 초기화를 수행"""

    def __init__(
        self,
        model_name_or_path: str = "openai-community/gpt2",
        original_tokenizer_path: Path = Path("artifacts/tokenizers/original"),
        remapped_tokenizer_path: Path = Path("artifacts/tokenizers/remapped"),
        remap_rules_path: Path = Path("src/gpt2_ivr/tokenizer/remap_rules.yaml"),
        embeddings_dir: Path = Path("artifacts/embeddings"),
    ) -> None:
        self.logger = get_logger("gpt2_ivr.align_command")
        self.model_name_or_path = model_name_or_path
        self.original_tokenizer_path = original_tokenizer_path
        self.remapped_tokenizer_path = remapped_tokenizer_path
        self.remap_rules_path = remap_rules_path
        self.embeddings_dir = embeddings_dir

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """커맨드 실행 로직"""
        self.logger.info("Executing Align Command...")

        # 1. 원본 모델에서 임베딩 추출
        self.logger.info("📊 1단계: 원본 모델 임베딩 추출")
        original_embeddings_path = self.embeddings_dir / "original_embeddings.pt"
        extract_embeddings(
            model_name_or_path=self.model_name_or_path,
            output_path=original_embeddings_path,
        )

        # 2. 재할당 규칙에 따라 임베딩 재정렬
        self.logger.info("🔄 2단계: 임베딩 재정렬")
        reordered_embeddings_path = self.embeddings_dir / "reordered_embeddings.pt"
        reorder_embeddings(
            original_embeddings_path=original_embeddings_path,
            original_tokenizer_path=self.original_tokenizer_path / "tokenizer.json",
            remapped_tokenizer_path=self.remapped_tokenizer_path / "tokenizer.json",
            remap_rules_path=self.remap_rules_path,
            output_path=reordered_embeddings_path,
        )

        # 3. 재정렬된 임베딩을 모델에 적용
        self.logger.info("🚀 3단계: 재정렬된 임베딩을 모델에 적용")
        initialized_model_path = self.embeddings_dir / "initialized_model"
        result = initialize_new_embeddings(
            reordered_embeddings_path=reordered_embeddings_path,
            model_save_path=initialized_model_path,
            model_name_or_path=self.model_name_or_path,
        )

        self.logger.info("✅ Align Command finished.")
        return {
            "status": "success",
            "original_embeddings": str(original_embeddings_path),
            "reordered_embeddings": str(reordered_embeddings_path),
            "initialized_model": str(initialized_model_path),
            "vocab_size": result.get("vocab_size"),
        }

    def get_name(self) -> str:
        """커맨드 이름 반환"""
        return "align"
