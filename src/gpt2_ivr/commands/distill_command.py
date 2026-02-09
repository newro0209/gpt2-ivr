"""토크나이저 증류 커맨드"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gpt2_ivr.tokenizer import distill_unigram_tokenizer

from .base import Command


class DistillCommand(Command):
    """토크나이저 증류 커맨드"""

    def __init__(
        self,
        original_tokenizer_dir: Path,
        distilled_tokenizer_dir: Path,
        corpus_dir: Path,
    ):
        self.original_tokenizer_dir = original_tokenizer_dir
        self.distilled_tokenizer_dir = distilled_tokenizer_dir
        self.corpus_dir = corpus_dir

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """토크나이저 증류 실행"""
        result = distill_unigram_tokenizer(
            original_tokenizer_dir=self.original_tokenizer_dir,
            distilled_tokenizer_dir=self.distilled_tokenizer_dir,
            corpus_dir=self.corpus_dir,
        )

        return {
            "output_dir": result["output_dir"],
            "vocab_size": result["vocab_size"],
            "original_vocab_size": result["original_vocab_size"],
        }

    def get_name(self) -> str:
        return "distill-tokenizer"
