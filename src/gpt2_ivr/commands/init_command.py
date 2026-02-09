"""모델 및 토크나이저 초기화 커맨드.

Hugging Face Hub에서 GPT-2 모델과 토크나이저를 다운로드하여
로컬에 저장하는 초기화 단계를 수행한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gpt2_ivr.tokenizer import initialize_assets

from .base import Command


class InitCommand(Command):
    """모델 및 토크나이저 초기화 커맨드.

    Hugging Face Hub에서 지정된 모델의 토크나이저를 다운로드하여
    로컬 디렉토리에 저장한다.

    Attributes:
        model_name: Hugging Face Hub 모델 이름
        tokenizer_dir: 토크나이저 저장 디렉토리
        force: 기존 파일이 있어도 재다운로드 여부
    """

    def __init__(
        self,
        model_name: str,
        tokenizer_dir: Path,
        force: bool,
    ):
        self.model_name = model_name
        self.tokenizer_dir = tokenizer_dir
        self.force = force

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """토크나이저 초기화를 실행한다.

        Args:
            **kwargs: 사용되지 않음

        Returns:
            초기화 결과 딕셔너리 (tokenizer_dir, vocab_size, model_name)
        """
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
        """커맨드 이름을 반환한다.

        Returns:
            커맨드 이름 "init"
        """
        return "init"
