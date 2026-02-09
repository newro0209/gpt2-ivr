"""IVR 교체 후보 선정 커맨드"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gpt2_ivr.analysis import select_replacement_candidates

from .base import Command


class SelectCommand(Command):
    """IVR 교체 후보 선정 커맨드"""

    def __init__(
        self,
        frequency_path: Path,
        sequences_path: Path,
        output_csv: Path,
        output_log: Path,
        model_name: str,
        max_candidates: int,
        min_token_len: int,
    ):
        self.frequency_path = frequency_path
        self.sequences_path = sequences_path
        self.output_csv = output_csv
        self.output_log = output_log
        self.model_name = model_name
        self.max_candidates = max_candidates
        self.min_token_len = min_token_len

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """교체 후보 선정 실행"""
        result = select_replacement_candidates(
            frequency_path=self.frequency_path,
            sequences_path=self.sequences_path,
            output_csv=self.output_csv,
            output_log=self.output_log,
            model_name=self.model_name,
            max_candidates=self.max_candidates,
            min_token_len=self.min_token_len,
        )

        return {
            "pairs_count": result["pairs_count"],
            "csv_path": result["csv_path"],
            "log_path": result["log_path"],
        }

    def get_name(self) -> str:
        return "select"
