"""모델 및 토크나이저 초기화 커맨드"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gpt2_ivr.tokenizer import initialize_assets

from .base import Command


class InitCommand(Command):
    """모델 및 토크나이저 초기화 커맨드"""

    def __init__(
        self,
        model_name: str = "openai-community/gpt2",
        tokenizer_dir: Path = Path("artifacts/tokenizers/original"),
        force: bool = False,
    ):
        self.model_name = model_name
        self.tokenizer_dir = tokenizer_dir
        self.force = force

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """초기화 실행"""
        result = initialize_assets(
            model_name=self.model_name,
            tokenizer_dir=self.tokenizer_dir,
            force=self.force,
        )

        return {
            "tokenizer_dir": result["tokenizer_dir"],
            "vocab_size": result["vocab_size"],
            "model_name": result["model_name"],
        }

    def get_name(self) -> str:
        return "init"
