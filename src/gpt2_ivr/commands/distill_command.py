"""토크나이저 증류 커맨드.

GPT-2 BPE 토크나이저를 Unigram 모델로 증류하는 단계를 수행한다.
원본 토크나이저의 어휘 크기를 유지하면서 코퍼스를 기반으로 학습한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from gpt2_ivr.constants import (
    CORPORA_CLEANED_DIR,
    TOKENIZER_DISTILLED_UNIGRAM_DIR,
    TOKENIZER_ORIGINAL_DIR,
)
from gpt2_ivr.parser import CliHelpFormatter

from .base import Command, SubparsersLike


class DistillCommand(Command):
    """토크나이저 증류 커맨드.

    BPE 토크나이저를 Unigram 모델로 증류하여 원본과 유사한 동작을 하지만
    확률 기반 토큰 분할이 가능한 토크나이저를 생성한다.

    Attributes:
        console: Rich 콘솔 인스턴스
        original_tokenizer_dir: 원본 BPE 토크나이저 디렉토리
        distilled_tokenizer_dir: 증류된 Unigram 토크나이저 저장 디렉토리
        corpus_dir: 학습에 사용할 코퍼스 디렉토리
    """

    @staticmethod
    def configure_parser(subparsers: SubparsersLike) -> None:
        """서브커맨드 파서를 설정한다.

        Args:
            subparsers: 서브파서 액션 객체
        """
        parser = subparsers.add_parser(
            "distill-tokenizer", help="BPE -> Unigram distillation", formatter_class=CliHelpFormatter
        )
        parser.add_argument(
            "--original-tokenizer-dir",
            type=Path,
            default=TOKENIZER_ORIGINAL_DIR,
            help="원본 토크나이저 디렉토리",
        )
        parser.add_argument(
            "--distilled-tokenizer-dir",
            type=Path,
            default=TOKENIZER_DISTILLED_UNIGRAM_DIR,
            help="증류된 토크나이저 디렉토리",
        )
        parser.add_argument("--corpus-dir", type=Path, default=CORPORA_CLEANED_DIR, help="학습 코퍼스 디렉토리")

    def __init__(
        self,
        console: Console,
        original_tokenizer_dir: Path,
        distilled_tokenizer_dir: Path,
        corpus_dir: Path,
    ):
        self.console = console
        self.original_tokenizer_dir = original_tokenizer_dir
        self.distilled_tokenizer_dir = distilled_tokenizer_dir
        self.corpus_dir = corpus_dir

    def execute(self) -> dict[str, Any]:
        """토크나이저 증류를 실행한다.

        Returns:
            증류 결과 딕셔너리 (output_dir, vocab_size, original_vocab_size)
        """
        from gpt2_ivr.tokenizer import distill_unigram_tokenizer
        result = distill_unigram_tokenizer(
            original_tokenizer_dir=self.original_tokenizer_dir,
            distilled_tokenizer_dir=self.distilled_tokenizer_dir,
            corpus_dir=self.corpus_dir,
        )

        # Rich 테이블로 결과 출력
        table = Table(title="🔬 토크나이저 증류 결과", show_header=True, title_style="bold green")
        table.add_column("항목", style="bold cyan", width=25)
        table.add_column("값", style="yellow", justify="right")

        table.add_row("원본 vocab 크기", f"{result['original_vocab_size']:,}개")
        table.add_row("증류 vocab 크기", f"{result['vocab_size']:,}개")
        vocab_diff = result["vocab_size"] - result["original_vocab_size"]
        diff_style = "green" if vocab_diff == 0 else "red" if vocab_diff < 0 else "yellow"
        table.add_row("차이", f"[{diff_style}]{vocab_diff:+,}개[/{diff_style}]")
        table.add_row("", "")  # 빈 줄
        table.add_row("저장 경로", str(result["output_dir"]))

        self.console.print()
        self.console.print(table)
        self.console.print()

        # 성공 메시지
        self.console.print(
            Panel(
                "[bold green]✅ Unigram 토크나이저 증류가 성공적으로 완료되었습니다[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )
        self.console.print()

        return {
            "output_dir": result["output_dir"],
            "vocab_size": result["vocab_size"],
            "original_vocab_size": result["original_vocab_size"],
        }

    def get_name(self) -> str:
        """커맨드 이름을 반환한다.

        Returns:
            커맨드 이름 "distill-tokenizer"
        """
        return "distill-tokenizer"
