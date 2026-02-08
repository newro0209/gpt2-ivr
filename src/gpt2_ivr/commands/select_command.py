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
        frequency_path: Path = Path("artifacts/analysis/reports/token_frequency.parquet"),
        sequences_path: Path = Path("artifacts/analysis/reports/bpe_token_id_sequences.txt"),
        output_csv: Path = Path("artifacts/analysis/reports/replacement_candidates.csv"),
        output_log: Path = Path("artifacts/analysis/reports/selection_log.md"),
        model_name: str = "openai-community/gpt2",
        max_candidates: int = 1000,
        min_token_len: int = 2,
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
