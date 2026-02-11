"""토큰 빈도 분석 커맨드.

코퍼스 파일을 BPE 토크나이저로 토큰화하여 각 토큰의 출현 빈도를
분석하고 시퀀스 파일과 빈도 통계를 생성한다.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

from gpt2_ivr.constants import (
    BPE_TOKEN_ID_SEQUENCES_FILE,
    CORPORA_CLEANED_DIR,
    TOKENIZER_ORIGINAL_DIR,
    TOKEN_FREQUENCY_FILE,
)
from gpt2_ivr.parser import CliHelpFormatter, non_negative_int

from .base import Command, SubparsersLike

logger = logging.getLogger(__name__)


class AnalyzeCommand(Command):
    """토큰 빈도 분석 커맨드.

    코퍼스를 토큰화하여 BPE 토큰 ID 시퀀스와 빈도 통계를 생성한다.
    병렬 처리를 통해 대용량 코퍼스를 효율적으로 처리한다.

    Attributes:
        console: Rich 콘솔 인스턴스
        input_dir: 코퍼스 입력 디렉토리
        output_sequences: BPE 토큰 시퀀스 출력 경로
        output_frequency: 토큰 빈도 parquet 출력 경로
        tokenizer_dir: 원본 토크나이저 디렉토리
        workers: 스레드 워커 수 (0이면 CPU - 1)
        chunk_size: 스레드 청크 크기 (0이면 자동 설정)
        max_texts: 처리할 최대 텍스트 수 (0이면 전체)
        encoding: 입력 파일 인코딩
    """

    @staticmethod
    def configure_parser(subparsers: SubparsersLike) -> None:
        """서브커맨드 파서를 설정한다.

        Args:
            subparsers: 서브파서 액션 객체
        """
        parser = subparsers.add_parser("analyze", help="BPE 토큰 시퀀스 분석", formatter_class=CliHelpFormatter)
        parser.add_argument("--input-dir", type=Path, default=CORPORA_CLEANED_DIR, help="코퍼스 입력 디렉토리")
        parser.add_argument(
            "--output-sequences", type=Path, default=BPE_TOKEN_ID_SEQUENCES_FILE, help="BPE 토큰 시퀀스 출력 경로"
        )
        parser.add_argument(
            "--output-frequency", type=Path, default=TOKEN_FREQUENCY_FILE, help="토큰 빈도 parquet 출력 경로"
        )
        parser.add_argument(
            "--tokenizer-dir",
            type=Path,
            default=TOKENIZER_ORIGINAL_DIR,
            help="원본 토크나이저 디렉토리",
        )
        parser.add_argument("--workers", type=non_negative_int, default=0, help="스레드 워커 수 (0이면 CPU - 1)")
        parser.add_argument("--chunk-size", type=non_negative_int, default=0, help="스레드 청크 크기(0이면 자동 설정)")
        parser.add_argument("--max-texts", type=non_negative_int, default=0, help="처리할 최대 텍스트 수 (0이면 전체)")
        parser.add_argument("--encoding", default="utf-8", help="입력 파일 인코딩")

    def __init__(
        self,
        console: Console,
        input_dir: Path,
        output_sequences: Path,
        output_frequency: Path,
        tokenizer_dir: Path,
        workers: int,
        chunk_size: int,
        max_texts: int,
        encoding: str,
    ):
        """빈도 분석 커맨드를 생성한다.

        Args:
            console: Rich 콘솔 인스턴스
            input_dir: 코퍼스 입력 디렉토리
            output_sequences: BPE 토큰 시퀀스 출력 경로
            output_frequency: 토큰 빈도 parquet 출력 경로
            tokenizer_dir: 원본 토크나이저 디렉토리
            workers: 스레드 워커 수 (0이면 CPU - 1)
            chunk_size: 스레드 청크 크기 (0이면 자동 설정)
            max_texts: 처리할 최대 텍스트 수 (0이면 전체)
            encoding: 입력 파일 인코딩
        """
        self.console = console
        self.input_dir = input_dir
        self.output_sequences = output_sequences
        self.output_frequency = output_frequency
        self.tokenizer_dir = tokenizer_dir
        self.workers = workers
        self.chunk_size = chunk_size
        self.max_texts = max_texts
        self.encoding = encoding

    def execute(self) -> dict[str, Any]:
        """토큰 빈도 분석을 실행한다.

        코퍼스를 읽어 토큰화하고, 토큰 ID 시퀀스 파일과 빈도 통계 파일을 생성한다.

        Returns:
            분석 결과 딕셔너리 (sequences_path, frequency_path, total_tokens, unique_tokens)
        """

        from gpt2_ivr.analysis.token_frequency import (
            analyze_token_frequency,
            write_frequency_parquet,
        )

        encoded_chunks_iterator, tokenizer = analyze_token_frequency(
            input_dir=self.input_dir,
            inputs=[],
            output_frequency=self.output_frequency,
            tokenizer_dir=self.tokenizer_dir,
            workers=self.workers,
            chunk_size=self.chunk_size,
            max_texts=self.max_texts,
            encoding=self.encoding,
        )

        counter: Counter[int] = Counter()
        self.output_sequences.parent.mkdir(parents=True, exist_ok=True)
        with self.output_sequences.open("w", encoding="utf-8") as handle:
            for chunk_ids in track(encoded_chunks_iterator, description="토큰화 중"):
                for token_ids in chunk_ids:
                    counter.update(token_ids)
                    handle.write(" ".join(str(token_id) for token_id in token_ids))
                    handle.write("\n")

        # 결과물 저장 (빈도 parquet)
        self.output_frequency.parent.mkdir(parents=True, exist_ok=True)
        write_frequency_parquet(counter, self.output_frequency)

        total_tokens = sum(counter.values())
        unique_tokens = len(counter)

        # 추가 통계 계산
        top_10 = counter.most_common(10)
        avg_frequency = total_tokens / unique_tokens if unique_tokens > 0 else 0

        # Rich 테이블로 결과 출력
        table = Table(title="✨ 토큰 빈도 분석 결과", show_header=True, title_style="bold green")
        table.add_column("항목", style="bold cyan", width=20)
        table.add_column("값", style="yellow", justify="right")

        table.add_row("총 토큰 수", f"{total_tokens:,}개")
        table.add_row("고유 토큰 수", f"{unique_tokens:,}개")
        table.add_row("평균 빈도", f"{avg_frequency:.2f}회")
        table.add_row("", "")  # 빈 줄
        table.add_row("빈도 파일", str(self.output_frequency))
        table.add_row("시퀀스 파일", str(self.output_sequences))

        self.console.print()
        self.console.print(table)

        # 상위 10개 토큰 표시
        if top_10:
            top_table = Table(title="🏆 상위 10개 빈도 토큰", show_header=True, border_style="dim")
            top_table.add_column("순위", style="dim", width=6, justify="center")
            top_table.add_column("토큰 ID", style="cyan", width=10, justify="right")
            top_table.add_column("빈도", style="yellow", width=15, justify="right")

            for idx, (token_id, freq) in enumerate(top_10, 1):
                rank_style = "bold green" if idx <= 3 else "dim"
                top_table.add_row(f"{idx}", f"{token_id}", f"{freq:,}회", style=rank_style)

            self.console.print()
            self.console.print(top_table)

        self.console.print()

        return {
            "sequences_path": self.output_sequences,
            "frequency_path": self.output_frequency,
            "total_tokens": total_tokens,
            "unique_tokens": unique_tokens,
        }

    def get_name(self) -> str:
        """커맨드 이름을 반환한다.

        Returns:
            커맨드 이름 "analyze"
        """
        return "analyze"
