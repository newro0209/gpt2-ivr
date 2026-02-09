"""IVR 교체 후보 선정 커맨드.

토큰 빈도 분석 결과를 기반으로 저빈도 희생 토큰과 고빈도 바이그램 병합 후보를
매칭하여 IVR(Infrequent Vocabulary Replacement) 교체 후보를 선정한다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rich.progress import track

from gpt2_ivr.analysis.candidate_selection import (
    select_replacement_candidates,
    write_replacement_csv,
    write_selection_log,
)

from .base import Command

logger = logging.getLogger(__name__)


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

        # 5) 바이그램 집계 (트래킹 추가)
        # select_replacement_candidates 내부에서 bigram_counts가 이미 계산되었으므로,
        # 여기서는 단순히 logger.info만 출력합니다.
        # track은 이미 도메인 함수에서 제거되었으므로, 여기서 track을 감쌀 필요는 없습니다.
        # bigram_counts는 Counter 객체이므로 iterable이 아닙니다.
        logger.info("고유 바이그램 %d개를 집계했습니다.", len(bigram_counts))

        # 6) 신규 토큰 후보 탐색 (트래킹 추가)
        # new_tokens_list도 이미 계산된 리스트이므로, 여기서 track을 감쌀 필요는 없습니다.
        logger.info("신규 토큰 후보 %d개를 선정했습니다.", len(new_tokens_list))

        # 8) 결과 저장
        write_replacement_csv(pairs, self.output_csv)
        logger.info("📄 교체 후보 CSV 저장 완료: %s", self.output_csv)

        write_selection_log(
            pairs=pairs,
            total_vocab=tokenizer.vocab_size,
            total_protected=len(
                sacrifices
            ),  # Corrected from protected_ids to sacrifices
            total_sacrifice_pool=tokenizer.vocab_size - len(sacrifices),  # Corrected
            total_bigrams=len(bigram_counts),
            output_path=self.output_log,
        )
        logger.info("📝 선정 로그 저장 완료: %s", self.output_log)

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
