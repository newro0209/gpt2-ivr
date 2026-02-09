"""IVR 교체 후보 토큰 선정 모듈.

token_frequency.parquet 와 bpe_token_id_sequences.txt 를 기반으로
저빈도 희생 토큰과 고빈도 도메인 바이그램 병합 후보를 매칭한다.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import TypedDict, cast

import pyarrow.parquet as pq
from transformers import GPT2Tokenizer

logger = logging.getLogger(__name__)


class SelectionResult(TypedDict):
    """선정 결과 타입.

    Attributes:
        pairs_count: 교체 후보 쌍 개수
        sacrifice_count: 희생 후보 개수
        new_token_count: 신규 토큰 후보 개수
        csv_path: CSV 파일 경로
        log_path: 로그 파일 경로
    """

    pairs_count: int
    sacrifice_count: int
    new_token_count: int
    csv_path: Path
    log_path: Path


@dataclass(frozen=True, slots=True)
class SacrificeCandidate:
    """희생(저빈도) 후보 토큰.

    Attributes:
        token_id: 토큰 ID
        token_str: 토큰 문자열 (디코딩 결과)
        frequency: 코퍼스 출현 빈도
    """

    token_id: int
    token_str: str
    frequency: int


@dataclass(frozen=True, slots=True)
class NewTokenCandidate:
    """신규(바이그램 병합) 토큰 후보.

    Attributes:
        merged_str: 병합된 문자열
        left_id: 왼쪽 토큰 ID
        right_id: 오른쪽 토큰 ID
        bigram_freq: 바이그램 출현 빈도
    """

    merged_str: str
    left_id: int
    right_id: int
    bigram_freq: int


@dataclass(frozen=True, slots=True)
class ReplacementPair:
    """교체 후보 쌍: 희생 토큰 → 신규 토큰.

    Attributes:
        rank: 순위 (1부터 시작)
        sacrifice: 희생 후보 토큰
        new_token: 신규 토큰 후보
        score: 교체 가치 점수
    """

    rank: int
    sacrifice: SacrificeCandidate
    new_token: NewTokenCandidate
    score: float


def load_frequency(path: Path) -> dict[int, int]:
    """token_frequency.parquet 에서 {token_id: frequency} 사전을 로드한다.

    Args:
        path: Parquet 파일 경로

    Returns:
        토큰 ID를 키로, 빈도를 값으로 하는 딕셔너리
    """
    table = pq.read_table(path, columns=["token_id", "frequency"])
    token_ids: list[int] = table.column("token_id").to_pylist()
    frequencies: list[int] = table.column("frequency").to_pylist()
    return dict(zip(token_ids, frequencies))


def get_protected_token_ids(
    tokenizer: GPT2Tokenizer,
    min_token_len: int,
) -> set[int]:
    """보호 대상 토큰 id 집합을 구성한다.

    보호 대상:
        - 스페셜 토큰 (``<|endoftext|>`` 등)
        - 디코딩 시 *min_token_len* 미만 문자열로 변환되는 토큰
          (바이트 수준 단일 문자 토큰 포함)

    Args:
        tokenizer: GPT-2 토크나이저
        min_token_len: 보호 대상 최소 길이 (이 길이 미만은 보호)

    Returns:
        보호 대상 토큰 ID 집합
    """
    protected: set[int] = set()

    # 1) 스페셜 토큰 보호
    for token_id in tokenizer.all_special_ids:
        protected.add(token_id)

    # 2) 짧은 토큰 보호 (바이트 수준 토큰 포함)
    vocab_size: int = tokenizer.vocab_size
    for token_id in range(vocab_size):
        decoded = cast(str, tokenizer.decode([token_id]))
        if len(decoded) < min_token_len:
            protected.add(token_id)

    return protected


def select_sacrifice_candidates(
    freq: dict[int, int],
    tokenizer: GPT2Tokenizer,
    protected_ids: set[int],
    max_candidates: int,
) -> list[SacrificeCandidate]:
    """저빈도 희생 후보 토큰을 선정한다.

    보호 대상을 제외한 전체 vocab 에서 빈도가 낮은 순으로 정렬하여
    상위 *max_candidates* 개를 반환한다. 빈도 0(코퍼스에 미출현)인 토큰이 최우선.

    Args:
        freq: 토큰 ID별 빈도 딕셔너리
        tokenizer: GPT-2 토크나이저
        protected_ids: 보호 대상 토큰 ID 집합
        max_candidates: 최대 후보 개수

    Returns:
        빈도 오름차순 정렬된 희생 후보 리스트
    """
    vocab_size: int = tokenizer.vocab_size
    candidates: list[SacrificeCandidate] = []

    for token_id in range(vocab_size):
        if token_id in protected_ids:
            continue
        token_str = cast(str, tokenizer.decode([token_id]))
        frequency = freq.get(token_id, 0)
        candidates.append(SacrificeCandidate(token_id, token_str, frequency))

    # 빈도 오름차순 → 동일 빈도 시 token_id 오름차순
    candidates.sort(key=lambda c: (c.frequency, c.token_id))
    return candidates[:max_candidates]


def count_bigrams(
    sequences_path: Path,
    logger: logging.Logger,
) -> Counter[tuple[int, int]]:
    """bpe_token_id_sequences.txt 에서 인접 토큰 바이그램 빈도를 집계한다.

    Args:
        sequences_path: 토큰 시퀀스 파일 경로
        logger: 로거 인스턴스

    Returns:
        (left_id, right_id) 튜플을 키로 하는 빈도 카운터
    """
    counter: Counter[tuple[int, int]] = Counter()

    with sequences_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                continue
            ids = [int(p) for p in parts]
            for i in range(len(ids) - 1):
                counter[(ids[i], ids[i + 1])] += 1

    logger.info("고유 바이그램 %d개를 집계했습니다.", len(counter))
    return counter


def discover_new_token_candidates(
    bigram_counts: Counter[tuple[int, int]],
    tokenizer: GPT2Tokenizer,
    max_candidates: int,
    logger: logging.Logger,
) -> list[NewTokenCandidate]:
    """바이그램 빈도에서 신규 토큰 후보를 추출한다.

    바이그램을 디코딩하여 병합 문자열을 생성하고,
    해당 문자열이 이미 단일 토큰으로 존재하는 경우는 제외한다.

    Args:
        bigram_counts: 바이그램 빈도 카운터
        tokenizer: GPT-2 토크나이저
        max_candidates: 최대 후보 개수
        logger: 로거 인스턴스

    Returns:
        빈도 내림차순 정렬된 신규 토큰 후보 리스트
    """
    check_limit = max_candidates * 10
    top_bigrams = bigram_counts.most_common(check_limit)

    seen_merged: set[str] = set()
    candidates: list[NewTokenCandidate] = []

    # 1) 상위 바이그램 순회하며 후보 선정
    # 2) 병합 문자열이 비어있거나 중복이거나 이미 단일 토큰인 경우 제외
    for (left_id, right_id), freq in top_bigrams:
        if len(candidates) >= max_candidates:
            break

        merged_str = cast(str, tokenizer.decode([left_id, right_id]))

        if not merged_str.strip():
            continue

        if merged_str in seen_merged:
            continue

        encoded: list[int] = tokenizer.encode(merged_str, add_special_tokens=False)
        if len(encoded) <= 1:
            continue

        seen_merged.add(merged_str)
        candidates.append(
            NewTokenCandidate(
                merged_str=merged_str,
                left_id=left_id,
                right_id=right_id,
                bigram_freq=freq,
            )
        )

    logger.info("신규 토큰 후보 %d개를 선정했습니다.", len(candidates))
    return candidates


def match_candidates(
    sacrifices: list[SacrificeCandidate],
    new_tokens: list[NewTokenCandidate],
) -> list[ReplacementPair]:
    """희생 후보와 신규 토큰 후보를 1:1 순위 매칭한다.

    점수 = ``bigram_freq / (sacrifice_freq + 1)`` — 높을수록 교체 가치가 크다.

    Args:
        sacrifices: 희생 후보 리스트 (빈도 오름차순 정렬)
        new_tokens: 신규 토큰 후보 리스트 (빈도 내림차순 정렬)

    Returns:
        교체 후보 쌍 리스트
    """
    count = min(len(sacrifices), len(new_tokens))
    pairs: list[ReplacementPair] = []

    for i in range(count):
        sacrifice = sacrifices[i]
        new_token = new_tokens[i]
        score = new_token.bigram_freq / (sacrifice.frequency + 1)
        pairs.append(
            ReplacementPair(
                rank=i + 1,
                sacrifice=sacrifice,
                new_token=new_token,
                score=score,
            )
        )

    return pairs


def write_replacement_csv(
    pairs: list[ReplacementPair],
    output_path: Path,
) -> None:
    """replacement_candidates.csv 를 저장한다.

    Args:
        pairs: 교체 후보 쌍 리스트
        output_path: CSV 파일 저장 경로
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank",
        "sacrifice_token_id",
        "sacrifice_token",
        "sacrifice_freq",
        "new_token",
        "new_token_left_id",
        "new_token_right_id",
        "bigram_freq",
        "score",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(
                {
                    "rank": pair.rank,
                    "sacrifice_token_id": pair.sacrifice.token_id,
                    "sacrifice_token": pair.sacrifice.token_str,
                    "sacrifice_freq": pair.sacrifice.frequency,
                    "new_token": pair.new_token.merged_str,
                    "new_token_left_id": pair.new_token.left_id,
                    "new_token_right_id": pair.new_token.right_id,
                    "bigram_freq": pair.new_token.bigram_freq,
                    "score": f"{pair.score:.4f}",
                }
            )


def write_selection_log(
    pairs: list[ReplacementPair],
    total_vocab: int,
    total_protected: int,
    total_sacrifice_pool: int,
    total_bigrams: int,
    output_path: Path,
) -> None:
    """selection_log.md 를 저장한다.

    Args:
        pairs: 교체 후보 쌍 리스트
        total_vocab: 전체 어휘 크기
        total_protected: 보호 토큰 개수
        total_sacrifice_pool: 희생 후보 풀 크기
        total_bigrams: 고유 바이그램 개수
        output_path: 마크다운 로그 파일 경로
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# IVR 교체 후보 선정 로그\n",
        "",
        "## 요약 통계\n",
        "",
        "| 항목 | 값 |",
        "|------|------|",
        f"| 전체 vocab 크기 | {total_vocab:,} |",
        f"| 보호 토큰 수 | {total_protected:,} |",
        f"| 희생 후보 풀 크기 | {total_sacrifice_pool:,} |",
        f"| 고유 바이그램 수 | {total_bigrams:,} |",
        f"| 최종 교체 후보 쌍 | {len(pairs):,} |",
        "",
        "## 선정 기준\n",
        "",
        "- **희생 후보**: 보호 대상 제외 후 코퍼스 빈도가 가장 낮은 토큰",
        "- **신규 후보**: 인접 토큰 바이그램 빈도가 가장 높으면서 "
        "단일 토큰으로 존재하지 않는 병합 문자열",
        "- **점수**: `bigram_freq / (sacrifice_freq + 1)` — 높을수록 교체 가치가 큼",
        "",
        "## 상위 교체 후보\n",
        "",
        "| 순위 | 희생 토큰 (id) | 희생 빈도 | 신규 토큰 | 바이그램 빈도 | 점수 |",
        "|------|----------------|-----------|-----------|---------------|------|",
    ]

    display_count = min(len(pairs), 50)
    for pair in pairs[:display_count]:
        sac_str = pair.sacrifice.token_str.replace("|", "\\|")
        new_str = pair.new_token.merged_str.replace("|", "\\|")
        lines.append(
            f"| {pair.rank} "
            f"| `{sac_str}` ({pair.sacrifice.token_id}) "
            f"| {pair.sacrifice.frequency:,} "
            f"| `{new_str}` "
            f"| {pair.new_token.bigram_freq:,} "
            f"| {pair.score:,.2f} |"
        )

    if len(pairs) > display_count:
        lines.append("")
        lines.append(f"> 전체 {len(pairs)}쌍 중 상위 {display_count}쌍만 표시합니다.")

    lines.append("")

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def select_replacement_candidates(
    frequency_path: Path,
    sequences_path: Path,
    output_csv: Path,
    output_log: Path,
    tokenizer_dir: Path,
    max_candidates: int,
    min_token_len: int,
) -> tuple[
    Counter[tuple[int, int]],
    list[NewTokenCandidate],
    GPT2Tokenizer,
    list[SacrificeCandidate],
    list[ReplacementPair],
]:
    """IVR 교체 후보를 선정한다.

    Args:
        frequency_path: 토큰 빈도 parquet 파일 경로
        sequences_path: BPE 토큰 시퀀스 파일 경로
        output_csv: 교체 후보 CSV 저장 경로 (SelectCommand에서 사용)
        output_log: 선정 로그 저장 경로 (SelectCommand에서 사용)
        tokenizer_dir: 원본 토크나이저 디렉토리
        max_candidates: 최대 후보 개수
        min_token_len: 보호 토큰 최소 길이

    Returns:
        필요한 집계 데이터 튜플: bigram_counts, new_tokens, tokenizer, sacrifices, pairs

    Raises:
        FileNotFoundError: 입력 파일 또는 원본 토크나이저가 없는 경우
    """
    # 입력 파일 검증
    if not frequency_path.exists():
        raise FileNotFoundError(f"토큰 빈도 파일이 없습니다: {frequency_path}")
    if not sequences_path.exists():
        raise FileNotFoundError(f"토큰 시퀀스 파일이 없습니다: {sequences_path}")

    # 1) 빈도 로드
    logger.info("📊 토큰 빈도를 로드합니다: %s", frequency_path)
    freq = load_frequency(frequency_path)
    logger.info("빈도 데이터 로드 완료 (고유 토큰 %d개)", len(freq))

    # 2) 토크나이저 로드
    tokenizer_files = list(tokenizer_dir.glob("*")) if tokenizer_dir.exists() else []
    has_tokenizer_files = any(
        f.name in ["tokenizer.json", "vocab.json", "merges.txt"]
        for f in tokenizer_files
    )
    if not has_tokenizer_files:
        raise FileNotFoundError(f"원본 토크나이저 파일이 없습니다: {tokenizer_dir}")

    logger.info("🔤 GPT-2 토크나이저를 로드합니다: %s", tokenizer_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(str(tokenizer_dir))

    # 3) 보호 토큰 집합 구성
    protected_ids = get_protected_token_ids(tokenizer, min_token_len)
    logger.info("🛡️ 보호 대상 토큰 %d개를 설정했습니다.", len(protected_ids))

    # 4) 희생 후보 선정
    logger.info("📉 희생 후보를 선정합니다 (최대 %d개)...", max_candidates)
    sacrifices = select_sacrifice_candidates(
        freq, tokenizer, protected_ids, max_candidates
    )
    logger.info("희생 후보 %d개를 선정했습니다.", len(sacrifices))
    if sacrifices:
        zero_freq_count = sum(1 for s in sacrifices if s.frequency == 0)
        logger.info("  └─ 미출현(빈도 0) 토큰: %d개", zero_freq_count)

    # 5) 바이그램 집계
    logger.info("🔍 인접 토큰 바이그램을 집계합니다...")
    bigram_counts = count_bigrams(sequences_path, logger)

    # 6) 신규 토큰 후보 탐색
    logger.info("🧩 신규 토큰 후보를 탐색합니다 (최대 %d개)...", max_candidates)
    new_tokens = discover_new_token_candidates(
        bigram_counts, tokenizer, max_candidates, logger
    )

    # 7) 매칭
    pairs = match_candidates(sacrifices, new_tokens)
    logger.info("✅ 교체 후보 %d쌍을 매칭했습니다.", len(pairs))

    return (
        bigram_counts,
        new_tokens,
        tokenizer,
        sacrifices,
        pairs,
    )
