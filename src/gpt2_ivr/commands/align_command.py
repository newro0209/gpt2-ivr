"""임베딩 정렬 커맨드.

원본 모델에서 임베딩을 추출하고, 재할당 규칙에 따라 재정렬한 후
신규 토큰의 임베딩을 초기화하는 3단계 프로세스를 수행한다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from gpt2_ivr.commands.base import Command
from gpt2_ivr.embedding import (
    extract_embeddings,
    initialize_new_token_embeddings,
    reorder_embeddings,
)

console = Console()


class AlignCommand(Command):
    """임베딩 추출, 재정렬 및 초기화를 수행한다.

    1단계: 원본 모델 임베딩 추출 (wte, wpe)
    2단계: 재할당 규칙에 따라 임베딩 재정렬
    3단계: 신규 토큰 임베딩 초기화 (mean, random, zeros)

    Attributes:
        logger: 로거 인스턴스
        model_name: GPT-2 모델 이름
        original_tokenizer_dir: 원본 토크나이저 디렉토리
        remapped_tokenizer_dir: 재할당 토크나이저 디렉토리
        remap_rules_path: 재할당 규칙 YAML 파일 경로
        embeddings_output_dir: 임베딩 출력 디렉토리
        init_strategy: 신규 토큰 임베딩 초기화 전략
    """

    def __init__(
        self,
        model_name: str,
        original_tokenizer_dir: Path,
        remapped_tokenizer_dir: Path,
        remap_rules_path: Path,
        embeddings_output_dir: Path,
        init_strategy: str,
    ) -> None:
        self.logger = logging.getLogger("gpt2_ivr.align")
        self.model_name = model_name
        self.original_tokenizer_dir = original_tokenizer_dir
        self.remapped_tokenizer_dir = remapped_tokenizer_dir
        self.remap_rules_path = remap_rules_path
        self.embeddings_output_dir = embeddings_output_dir
        self.init_strategy = init_strategy

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """임베딩 정렬을 실행한다.

        임베딩 추출, 재정렬, 초기화의 3단계를 순차적으로 수행한다.

        Args:
            **kwargs: 사용되지 않음

        Returns:
            실행 결과 딕셔너리 (status, extract_result, reorder_result, init_result, embeddings_dir)
        """
        # 1. 원본 모델에서 임베딩 추출
        console.print()
        console.print(Panel("[bold cyan]1단계: 원본 모델 임베딩 추출[/bold cyan]", border_style="blue"))

        extract_result = extract_embeddings(
            model_name=self.model_name,
            output_dir=self.embeddings_output_dir,
            logger=self.logger,
        )

        # 2. Remap 규칙에 따라 임베딩 재정렬
        console.print()
        console.print(Panel("[bold cyan]2단계: 임베딩 재정렬[/bold cyan]", border_style="blue"))

        reorder_result = reorder_embeddings(
            original_wte_path=extract_result["wte"],
            original_tokenizer_dir=self.original_tokenizer_dir,
            remapped_tokenizer_dir=self.remapped_tokenizer_dir,
            remap_rules_path=self.remap_rules_path,
            output_dir=self.embeddings_output_dir,
            logger=self.logger,
        )

        # 3. 신규 토큰 임베딩 초기화
        console.print()
        console.print(Panel("[bold cyan]3단계: 신규 토큰 임베딩 초기화[/bold cyan]", border_style="blue"))

        init_result = initialize_new_token_embeddings(
            aligned_wte_path=reorder_result["aligned_wte"],
            original_tokenizer_dir=self.original_tokenizer_dir,
            remapped_tokenizer_dir=self.remapped_tokenizer_dir,
            remap_rules_path=self.remap_rules_path,
            output_dir=self.embeddings_output_dir,
            init_strategy=self.init_strategy,
            logger=self.logger,
        )

        # 결과 요약 출력
        result_text = f"""[bold green]임베딩 정렬 완료[/bold green]

[yellow]임베딩 디렉토리:[/yellow] {self.embeddings_output_dir}
[yellow]초기화 전략:[/yellow] {self.init_strategy}"""

        console.print()
        console.print(Panel(result_text, title="align 단계 완료", border_style="green"))
        console.print()

        return {
            "status": "success",
            "extract_result": extract_result,
            "reorder_result": reorder_result,
            "init_result": init_result,
            "embeddings_dir": str(self.embeddings_output_dir),
        }

    def get_name(self) -> str:
        """커맨드 이름을 반환한다.

        Returns:
            커맨드 이름 "align"
        """
        return "align"
