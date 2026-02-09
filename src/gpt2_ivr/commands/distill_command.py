"""토크나이저 증류 커맨드.

GPT-2 BPE 토크나이저를 Unigram 모델로 증류하는 단계를 수행한다.
원본 토크나이저의 어휘 크기를 유지하면서 코퍼스를 기반으로 학습한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gpt2_ivr.tokenizer import distill_unigram_tokenizer

from .base import Command


class DistillCommand(Command):
    """토크나이저 증류 커맨드.

    BPE 토크나이저를 Unigram 모델로 증류하여 원본과 유사한 동작을 하지만
    확률 기반 토큰 분할이 가능한 토크나이저를 생성한다.

    Attributes:
        original_tokenizer_dir: 원본 BPE 토크나이저 디렉토리
        distilled_tokenizer_dir: 증류된 Unigram 토크나이저 저장 디렉토리
        corpus_dir: 학습에 사용할 코퍼스 디렉토리
    """

    def __init__(
        self,
        original_tokenizer_dir: Path,
        distilled_tokenizer_dir: Path,
        corpus_dir: Path,
    ):
        self.original_tokenizer_dir = original_tokenizer_dir
        self.distilled_tokenizer_dir = distilled_tokenizer_dir
        self.corpus_dir = corpus_dir

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """토크나이저 증류를 실행한다.

        Args:
            **kwargs: 사용되지 않음

        Returns:
            증류 결과 딕셔너리 (output_dir, vocab_size, original_vocab_size)
        """
        result = distill_unigram_tokenizer(
            original_tokenizer_dir=self.original_tokenizer_dir,
            distilled_tokenizer_dir=self.distilled_tokenizer_dir,
            corpus_dir=self.corpus_dir,
        )

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
