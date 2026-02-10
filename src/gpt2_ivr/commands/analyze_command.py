"""토큰 빈도 분석 커맨드.

코퍼스 파일을 BPE 토크나이저로 토큰화하여 각 토큰의 출현 빈도를
분석하고 시퀀스 파일과 빈도 통계를 생성한다.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

from gpt2_ivr.analysis.token_frequency import (
    analyze_token_frequency,
    write_frequency_parquet,
)

from .base import Command

logger = logging.getLogger(__name__)
console = Console()


class AnalyzeCommand(Command):
    """토큰 빈도 분석 커맨드.

    코퍼스를 토큰화하여 BPE 토큰 ID 시퀀스와 빈도 통계를 생성한다.
    병렬 처리를 통해 대용량 코퍼스를 효율적으로 처리한다.

    Attributes:
        input_dir: 코퍼스 입력 디렉토리
        output_sequences: BPE 토큰 시퀀스 출력 경로
        output_frequency: 토큰 빈도 parquet 출력 경로
        tokenizer_dir: 원본 토크나이저 디렉토리
        workers: 스레드 워커 수 (0이면 CPU - 1)
        chunk_size: 스레드 청크 크기 (0이면 자동 설정)
        max_texts: 처리할 최대 텍스트 수 (0이면 전체)
        encoding: 입력 파일 인코딩
    """

    def __init__(
        self,
        input_dir: Path,
        output_sequences: Path,
        output_frequency: Path,
        tokenizer_dir: Path,
        workers: int,
        chunk_size: int,
        max_texts: int,
        encoding: str,
    ):
        self.input_dir = input_dir
        self.output_sequences = output_sequences
        self.output_frequency = output_frequency
        self.tokenizer_dir = tokenizer_dir
        self.workers = workers
        self.chunk_size = chunk_size
        self.max_texts = max_texts
        self.encoding = encoding

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """토큰 빈도 분석을 실행한다.

        코퍼스를 읽어 토큰화하고, 토큰 ID 시퀀스 파일과 빈도 통계 파일을 생성한다.

        Args:
            **kwargs: 사용되지 않음

        Returns:
            분석 결과 딕셔너리 (sequences_path, frequency_path, total_tokens, unique_tokens)
        """
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

        # Rich 테이블로 결과 출력
        table = Table(title="토큰 빈도 분석 결과", show_header=False, title_style="bold green")
        table.add_column("항목", style="cyan", width=20)
        table.add_column("값", style="yellow")

        table.add_row("총 토큰", f"{total_tokens:,}개")
        table.add_row("고유 토큰", f"{unique_tokens:,}개")
        table.add_row("빈도 파일", str(self.output_frequency))
        table.add_row("시퀀스 파일", str(self.output_sequences))

        console.print()
        console.print(table)
        console.print()

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
