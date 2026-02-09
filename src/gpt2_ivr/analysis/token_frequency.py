from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import os
from pathlib import Path
from threading import Lock
from typing import Iterable, Iterator, TypedDict, cast

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import BatchEncoding, GPT2Tokenizer

from gpt2_ivr.utils.logging_config import get_logger

logger = get_logger(__name__)


class FrequencyResult(TypedDict):
    """빈도 분석 결과 타입"""

    total_tokens: int
    unique_tokens: int
    sequences_path: Path
    frequency_path: Path


def find_input_files(input_dir: Path, inputs: list[Path]) -> list[Path]:
    """분석 대상 파일 목록을 수집한다."""
    allowed_suffixes = {".txt", ".jsonl", ".json"}
    files: list[Path] = []

    if input_dir.exists():
        for path in sorted(input_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in allowed_suffixes:
                files.append(path)

    for path in inputs:
        if path.is_file():
            files.append(path)

    return sorted(set(files))


def iter_texts(files: list[Path], text_key: str, encoding: str) -> Iterator[str]:
    """파일에서 텍스트 스트림을 생성한다."""
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            with path.open("r", encoding=encoding) as handle:
                for line in handle:
                    text = line.rstrip("\n")
                    if text.strip():
                        yield text
            continue

        if suffix == ".jsonl":
            with path.open("r", encoding=encoding) as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if text_key in record and isinstance(record[text_key], str):
                        text = record[text_key].strip()
                        if text:
                            yield text
            continue

        if suffix == ".json":
            with path.open("r", encoding=encoding) as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                for record in payload:
                    if isinstance(record, dict) and text_key in record:
                        text = str(record[text_key]).strip()
                        if text:
                            yield text
            elif isinstance(payload, dict) and text_key in payload:
                text = str(payload[text_key]).strip()
                if text:
                    yield text


def write_frequency_parquet(counter: Counter[int], output_path: Path) -> None:
    """토큰 빈도를 parquet로 저장한다."""
    rows = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    token_ids = [token_id for token_id, _ in rows]
    frequencies = [frequency for _, frequency in rows]
    table = pa.Table.from_pydict({"token_id": token_ids, "frequency": frequencies})
    pq.write_table(table, output_path)


def collect_statistics(
    texts: Iterable[str],
    output_sequences: Path,
    tokenizer: GPT2Tokenizer,
    workers: int,
    chunk_size: int,
) -> Counter[int]:
    """토큰 시퀀스와 빈도 통계를 생성한다."""
    counter: Counter[int] = Counter()
    encode_lock = Lock()

    # 입력 스트림 청크 분할
    def chunk_iter(source: Iterable[str]) -> Iterator[list[str]]:
        iterator = iter(source)
        while True:
            chunk = list(islice(iterator, chunk_size))
            if not chunk:
                break
            yield chunk

    # 청크 단위 토큰화
    def encode_chunk(chunk: list[str]) -> list[list[int]]:
        # 토크나이저 스레드 안전성 보호
        with encode_lock:
            output: BatchEncoding = tokenizer(
                chunk,
                add_special_tokens=False,
                truncation=True,
                max_length=tokenizer.model_max_length,
            )
        return cast(list[list[int]], output["input_ids"])

    output_sequences.parent.mkdir(parents=True, exist_ok=True)
    with output_sequences.open("w", encoding="utf-8") as handle:
        # 청크 스레드 병렬 처리
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for chunk_ids in tqdm(
                executor.map(encode_chunk, chunk_iter(texts)),
                desc="토큰화",
                unit="청크",
            ):
                # 청크 결과 누적 및 기록
                for token_ids in chunk_ids:
                    counter.update(token_ids)
                    handle.write(" ".join(str(token_id) for token_id in token_ids))
                    handle.write("\n")

    return counter


def analyze_token_frequency(
    input_dir: Path,
    inputs: list[Path],
    output_sequences: Path,
    output_frequency: Path,
    model_name: str,
    text_key: str,
    workers: int,
    chunk_size: int,
    max_texts: int,
    encoding: str,
) -> FrequencyResult:
    """토큰 빈도를 분석한다.

    Args:
        input_dir: 코퍼스 입력 디렉토리
        inputs: 개별 입력 파일 목록
        output_sequences: BPE 토큰 시퀀스 출력 경로
        output_frequency: 토큰 빈도 parquet 출력 경로
        model_name: GPT-2 모델 이름
        text_key: json/jsonl 텍스트 키
        workers: 스레드 워커 수 (0이면 CPU * 2)
        chunk_size: 스레드 청크 크기
        max_texts: 처리할 최대 텍스트 수 (0이면 전체)
        encoding: 입력 파일 인코딩

    Returns:
        분석 결과 정보를 담은 딕셔너리

    Raises:
        SystemExit: 입력 파일을 찾을 수 없는 경우
    """
    # 입력 파일 목록 수집
    input_files = find_input_files(input_dir, inputs)
    if not input_files:
        raise SystemExit("입력 파일을 찾을 수 없습니다.")

    logger.info("📂 입력 파일 %d개를 탐색했습니다.", len(input_files))

    # 텍스트 스트림 구성
    texts = iter_texts(input_files, text_key, encoding)
    if max_texts > 0:
        texts = islice(texts, max_texts)
        logger.info("⚠️  최대 %d개 텍스트만 처리합니다.", max_texts)

    # 토크나이저 로드
    logger.info("🔤 GPT-2 토크나이저를 로드합니다: %s", model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # 워커 수 계산
    if workers <= 0:
        workers = max(1, int((os.cpu_count() or 1) * 2))

    logger.info("🔧 토큰화 설정: workers=%d, chunk_size=%d", workers, chunk_size)

    # 토큰화 및 빈도 집계
    counter = collect_statistics(
        texts,
        output_sequences=output_sequences,
        tokenizer=tokenizer,
        workers=workers,
        chunk_size=chunk_size,
    )

    # 결과물 저장
    output_frequency.parent.mkdir(parents=True, exist_ok=True)
    write_frequency_parquet(counter, output_frequency)

    total_tokens = sum(counter.values())
    unique_tokens = len(counter)

    logger.info("✅ 토큰 빈도 분석 완료")
    logger.info("  └─ 총 토큰: %d개", total_tokens)
    logger.info("  └─ 고유 토큰: %d개", unique_tokens)
    logger.info("📄 토큰 빈도 저장: %s", output_frequency)
    logger.info("📄 토큰 시퀀스 저장: %s", output_sequences)

    return FrequencyResult(
        total_tokens=total_tokens,
        unique_tokens=unique_tokens,
        sequences_path=output_sequences,
        frequency_path=output_frequency,
    )
