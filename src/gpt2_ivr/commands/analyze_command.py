"""토큰 빈도 분석 커맨드"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gpt2_ivr.analysis import analyze_token_frequency
from gpt2_ivr.constants import (
    BPE_TOKEN_ID_SEQUENCES_FILE,
    CORPORA_CLEANED_DIR,
    TOKEN_FREQUENCY_FILE,
)

from .base import Command


class AnalyzeCommand(Command):
    """토큰 빈도 분석 커맨드"""

    def __init__(
        self,
        input_dir: Path = CORPORA_CLEANED_DIR,
        output_sequences: Path = BPE_TOKEN_ID_SEQUENCES_FILE,
        output_frequency: Path = TOKEN_FREQUENCY_FILE,
        model_name: str = "openai-community/gpt2",
        workers: int = 0,
        chunk_size: int = 50,
        max_texts: int = 0,
    ):
        self.input_dir = input_dir
        self.output_sequences = output_sequences
        self.output_frequency = output_frequency
        self.model_name = model_name
        self.workers = workers
        self.chunk_size = chunk_size
        self.max_texts = max_texts

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """토큰 빈도 분석 실행"""
        result = analyze_token_frequency(
            input_dir=self.input_dir,
            inputs=[],
            output_sequences=self.output_sequences,
            output_frequency=self.output_frequency,
            model_name=self.model_name,
            workers=self.workers,
            chunk_size=self.chunk_size,
            max_texts=self.max_texts,
        )

        return {
            "sequences_path": result["sequences_path"],
            "frequency_path": result["frequency_path"],
            "total_tokens": result["total_tokens"],
            "unique_tokens": result["unique_tokens"],
        }

    def get_name(self) -> str:
        return "analyze"
