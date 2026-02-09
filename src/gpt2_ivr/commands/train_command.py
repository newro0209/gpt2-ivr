"""Training Command"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gpt2_ivr.commands.base import Command
from gpt2_ivr.training.train import train_model
from gpt2_ivr.utils.logging_config import get_logger


class TrainCommand(Command):
    """Train Command"""

    def __init__(
        self,
        model_name_or_path: str = "openai-community/gpt2",
        tokenizer_path: Path = Path("artifacts/tokenizers/remapped"),
        dataset_path: Path = Path("artifacts/corpora/cleaned"),
        output_dir: Path = Path("artifacts/training/sft_checkpoint"),
        config_path: Path = Path("src/gpt2_ivr/training/sft_config.yaml"),
    ) -> None:
        self.logger = get_logger("gpt2_ivr.train_command")
        self.model_name_or_path = model_name_or_path
        self.tokenizer_path = tokenizer_path
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.config_path = config_path

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """커맨드 실행 로직"""
        self.logger.info("Executing Train Command...")

        # 학습 실행
        result = train_model(
            model_name_or_path=self.model_name_or_path,
            tokenizer_path=self.tokenizer_path,
            dataset_path=self.dataset_path,
            output_dir=self.output_dir,
            config_path=self.config_path,
        )

        self.logger.info("Train Command finished.")
        return result

    def get_name(self) -> str:
        """커맨드 이름 반환"""
        return "train"
