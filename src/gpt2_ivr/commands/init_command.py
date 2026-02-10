"""모델 및 토크나이저 초기화 커맨드.

Hugging Face Hub에서 GPT-2 모델과 토크나이저를 다운로드하여
로컬에 저장하는 초기화 단계를 수행한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from gpt2_ivr.corpus.normalize import normalize_raw_corpora
from gpt2_ivr.tokenizer import initialize_assets

from .base import Command

console = Console()


class InitCommand(Command):
    """모델 및 토크나이저 초기화 커맨드.

    Hugging Face Hub에서 지정된 모델의 토크나이저를 다운로드하여
    로컬 디렉토리에 저장한다.

    Attributes:
        model_name: Hugging Face Hub 모델 이름
        tokenizer_dir: 토크나이저 저장 디렉토리
        force: 기존 파일이 있어도 재다운로드 여부
        raw_corpora_dir: 정제 전 원본 코퍼스 디렉토리
        cleaned_corpora_dir: 정제된 코퍼스를 저장할 디렉토리
        text_key: JSON/JSONL 파일에서 추출할 텍스트 키
        encoding: 코퍼스 파일 인코딩
        normalize_force: 존재하는 정제본이 있어도 덮어쓸지 여부
    """

    def __init__(
        self,
        model_name: str,
        tokenizer_dir: Path,
        force: bool,
        raw_corpora_dir: Path,
        cleaned_corpora_dir: Path,
        text_key: str,
        encoding: str,
        normalize_force: bool,
    ):
        self.model_name = model_name
        self.tokenizer_dir = tokenizer_dir
        self.force = force
        self.raw_corpora_dir = raw_corpora_dir
        self.cleaned_corpora_dir = cleaned_corpora_dir
        self.text_key = text_key
        self.encoding = encoding
        self.normalize_force = normalize_force

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

        normalized_corpora = normalize_raw_corpora(
            raw_dir=self.raw_corpora_dir,
            cleaned_dir=self.cleaned_corpora_dir,
            text_key=self.text_key,
            encoding=self.encoding,
            force=self.normalize_force,
        )

        # Rich 테이블로 결과 출력
        table = Table(title="🚀 초기화 완료", show_header=True, title_style="bold green")
        table.add_column("항목", style="bold cyan", width=25)
        table.add_column("값", style="yellow", justify="left")

        table.add_row("모델", f"[bold]{result['model_name']}[/bold]")
        table.add_row("Vocab 크기", f"{result['vocab_size']:,}개")
        table.add_row("", "")  # 빈 줄
        table.add_row("토크나이저 경로", str(result["tokenizer_dir"]))
        table.add_row("정제된 코퍼스", f"{len(normalized_corpora):,}개 파일")
        table.add_row("코퍼스 경로", str(self.cleaned_corpora_dir))

        console.print()
        console.print(table)
        console.print()

        # 성공 메시지
        console.print(Panel(
            "[bold green]✅ 모델 및 코퍼스 초기화가 완료되었습니다[/bold green]\n"
            "[dim]다음 단계: [cyan]ivr analyze[/cyan] 명령으로 토큰 분석을 시작하세요[/dim]",
            border_style="green",
            padding=(1, 2)
        ))
        console.print()

        return {
            "tokenizer_dir": result["tokenizer_dir"],
            "vocab_size": result["vocab_size"],
            "model_name": result["model_name"],
            "normalized_corpora": len(normalized_corpora),
            "cleaned_dir": self.cleaned_corpora_dir,
        }

    def get_name(self) -> str:
        """커맨드 이름을 반환한다.

        Returns:
            커맨드 이름 "init"
        """
        return "init"
