"""IVR 교체 후보 선정 커맨드.

토큰 빈도 분석 결과를 기반으로 저빈도 희생 토큰과 고빈도 바이그램 병합 후보를
매칭하여 IVR(Infrequent Vocabulary Replacement) 교체 후보를 선정한다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from gpt2_ivr.analysis.candidate_selection import (
    select_replacement_candidates,
    write_replacement_csv,
    write_selection_log,
)

from .base import Command

logger = logging.getLogger(__name__)
console = Console()


class SelectCommand(Command):
    """IVR 교체 후보 선정 커맨드.

    저빈도 토큰을 희생 후보로, 고빈도 바이그램을 신규 토큰 후보로 선정하여
    교체 쌍을 생성한다.

    Attributes:
        frequency_path: 토큰 빈도 parquet 파일 경로
        sequences_path: BPE 토큰 시퀀스 파일 경로
        output_csv: 교체 후보 CSV 저장 경로
        output_log: 선정 로그 저장 경로
        tokenizer_dir: 원본 토크나이저 디렉토리
        max_candidates: 최대 후보 개수
        min_token_len: 보호 토큰 최소 길이
    """

    def __init__(
        self,
        frequency_path: Path,
        sequences_path: Path,
        output_csv: Path,
        output_log: Path,
        tokenizer_dir: Path,
        max_candidates: int,
        min_token_len: int,
    ):
        self.frequency_path = frequency_path
        self.sequences_path = sequences_path
        self.output_csv = output_csv
        self.output_log = output_log
        self.tokenizer_dir = tokenizer_dir
        self.max_candidates = max_candidates
        self.min_token_len = min_token_len

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """교체 후보 선정을 실행한다.

        토큰 빈도와 바이그램 통계를 분석하여 교체 후보 쌍을 생성하고
        CSV와 마크다운 로그로 저장한다.

        Args:
            **kwargs: 사용되지 않음

        Returns:
            선정 결과 딕셔너리 (pairs_count, csv_path, log_path, sacrifice_count, new_token_count)
        """
        (
            bigram_counts,
            new_tokens_list,
            tokenizer,
            sacrifices,
            pairs,
        ) = select_replacement_candidates(
            frequency_path=self.frequency_path,
            sequences_path=self.sequences_path,
            tokenizer_dir=self.tokenizer_dir,
            max_candidates=self.max_candidates,
            min_token_len=self.min_token_len,
        )

        # 결과 저장
        write_replacement_csv(pairs, self.output_csv)
        write_selection_log(
            pairs=pairs,
            total_vocab=tokenizer.vocab_size,
            total_protected=len(sacrifices),
            total_sacrifice_pool=tokenizer.vocab_size - len(sacrifices),
            total_bigrams=len(bigram_counts),
            output_path=self.output_log,
        )

        # Rich 테이블로 결과 출력
        table = Table(title="교체 후보 선정 결과", show_header=False, title_style="bold green")
        table.add_column("항목", style="cyan", width=20)
        table.add_column("값", style="yellow")

        table.add_row("교체 후보 쌍", f"{len(pairs):,}개")
        table.add_row("희생 후보", f"{len(sacrifices):,}개")
        table.add_row("신규 토큰 후보", f"{len(new_tokens_list):,}개")
        table.add_row("고유 바이그램", f"{len(bigram_counts):,}개")
        table.add_row("CSV 파일", str(self.output_csv))
        table.add_row("로그 파일", str(self.output_log))

        console.print()
        console.print(table)
        console.print()

        return {
            "pairs_count": len(pairs),
            "csv_path": self.output_csv,
            "log_path": self.output_log,
            "sacrifice_count": len(sacrifices),
            "new_token_count": len(new_tokens_list),
        }

    def get_name(self) -> str:
        """커맨드 이름을 반환한다.

        Returns:
            커맨드 이름 "select"
        """
        return "select"
