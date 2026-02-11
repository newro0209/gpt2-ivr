"""IVR 교체 후보 선정 커맨드.

토큰 빈도 분석 결과를 기반으로 저빈도 희생 토큰과 고빈도 바이그램 병합 후보를
매칭하여 IVR(Infrequent Vocabulary Replacement) 교체 후보를 선정한다.
"""

from __future__ import annotations

import argparse
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
from gpt2_ivr.constants import (
    BPE_TOKEN_ID_SEQUENCES_FILE,
    REPLACEMENT_CANDIDATES_FILE,
    SELECTION_LOG_FILE,
    TOKENIZER_ORIGINAL_DIR,
    TOKEN_FREQUENCY_FILE,
)
from gpt2_ivr.parser import CliHelpFormatter, positive_int

from .base import Command, SubparsersLike

logger = logging.getLogger(__name__)


class SelectCommand(Command):
    """IVR 교체 후보 선정 커맨드.

    저빈도 토큰을 희생 후보로, 고빈도 바이그램을 신규 토큰 후보로 선정하여
    교체 쌍을 생성한다.

    Attributes:
        console: Rich 콘솔 인스턴스
        frequency_path: 토큰 빈도 parquet 파일 경로
        sequences_path: BPE 토큰 시퀀스 파일 경로
        output_csv: 교체 후보 CSV 저장 경로
        output_log: 선정 로그 저장 경로
        tokenizer_dir: 원본 토크나이저 디렉토리
        max_candidates: 최대 후보 개수
        min_token_len: 보호 토큰 최소 길이
    """

    @staticmethod
    def configure_parser(subparsers: SubparsersLike) -> None:
        """서브커맨드 파서를 설정한다.

        Args:
            subparsers: 서브파서 액션 객체
        """
        parser = subparsers.add_parser("select", help="IVR 대상 토큰 선정", formatter_class=CliHelpFormatter)
        parser.add_argument(
            "--frequency-path", type=Path, default=TOKEN_FREQUENCY_FILE, help="토큰 빈도 parquet 파일 경로"
        )
        parser.add_argument(
            "--sequences-path", type=Path, default=BPE_TOKEN_ID_SEQUENCES_FILE, help="BPE 토큰 시퀀스 파일 경로"
        )
        parser.add_argument(
            "--output-csv", type=Path, default=REPLACEMENT_CANDIDATES_FILE, help="교체 후보 CSV 저장 경로"
        )
        parser.add_argument("--output-log", type=Path, default=SELECTION_LOG_FILE, help="선정 로그 저장 경로")
        parser.add_argument(
            "--tokenizer-dir",
            type=Path,
            default=TOKENIZER_ORIGINAL_DIR,
            help="원본 토크나이저 디렉토리",
        )
        parser.add_argument("--max-candidates", type=positive_int, default=1000, help="최대 후보 개수")
        parser.add_argument("--min-token-len", type=positive_int, default=2, help="보호 토큰 최소 길이")

    def __init__(
        self,
        console: Console,
        frequency_path: Path,
        sequences_path: Path,
        output_csv: Path,
        output_log: Path,
        tokenizer_dir: Path,
        max_candidates: int,
        min_token_len: int,
    ):
        self.console = console
        self.frequency_path = frequency_path
        self.sequences_path = sequences_path
        self.output_csv = output_csv
        self.output_log = output_log
        self.tokenizer_dir = tokenizer_dir
        self.max_candidates = max_candidates
        self.min_token_len = min_token_len

    def execute(self) -> dict[str, Any]:
        """교체 후보 선정을 실행한다.

        토큰 빈도와 바이그램 통계를 분석하여 교체 후보 쌍을 생성하고
        CSV와 마크다운 로그로 저장한다.

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
            output_csv=self.output_csv,
            output_log=self.output_log,
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
        table = Table(title="🎯 교체 후보 선정 결과", show_header=True, title_style="bold green")
        table.add_column("항목", style="bold cyan", width=20)
        table.add_column("값", style="yellow", justify="right")

        table.add_row("교체 후보 쌍", f"{len(pairs):,}개")
        table.add_row("희생 후보", f"{len(sacrifices):,}개")
        table.add_row("신규 토큰 후보", f"{len(new_tokens_list):,}개")
        table.add_row("고유 바이그램", f"{len(bigram_counts):,}개")
        table.add_row("", "")  # 빈 줄
        table.add_row("CSV 파일", str(self.output_csv))
        table.add_row("로그 파일", str(self.output_log))

        self.console.print()
        self.console.print(table)

        # 샘플 교체 후보 표시 (상위 5개)
        if pairs:
            sample_table = Table(title="📋 교체 후보 샘플 (상위 5개)", show_header=True, border_style="dim")
            sample_table.add_column("희생 토큰 ID", style="red", width=15, justify="center")
            sample_table.add_column("→", style="dim", width=3, justify="center")
            sample_table.add_column("신규 토큰", style="green", width=30)
            sample_table.add_column("빈도", style="yellow", width=12, justify="right")

            for pair in pairs[:5]:
                sacrifice_id = pair.sacrifice.token_id
                new_token = pair.new_token.merged_str
                frequency = pair.new_token.bigram_freq
                sample_table.add_row(f"{sacrifice_id}", "→", f"{new_token}", f"{frequency:,}회")

            self.console.print()
            self.console.print(sample_table)

        self.console.print()

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
