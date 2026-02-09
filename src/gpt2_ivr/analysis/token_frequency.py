"""토큰 빈도 분석 모듈.

코퍼스를 토큰화하여 각 토큰의 출현 빈도를 집계하고 Parquet 형식으로 저장한다.
병렬 처리를 통해 대용량 코퍼스를 효율적으로 처리한다.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import os
from pathlib import Path
from threading import Lock
from typing import Iterable, Iterator, TypedDict, cast

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import BatchEncoding, GPT2Tokenizer

logger = logging.getLogger(__name__)


class FrequencyResult(TypedDict):
    """빈도 분석 결과 타입.

    Attributes:
        total_tokens: 전체 토큰 개수
        unique_tokens: 고유 토큰 개수
        sequences_path: 토큰 시퀀스 파일 경로
        frequency_path: 빈도 parquet 파일 경로
    """

    total_tokens: int
    unique_tokens: int
    sequences_path: Path
    frequency_path: Path


def find_input_files(input_dir: Path, inputs: list[Path]) -> list[Path]:
    """분석 대상 파일 목록을 수집한다.

    input_dir에서 재귀적으로 파일을 탐색하고, inputs 리스트의 파일을 추가한다.
    .txt, .jsonl, .json 확장자만 허용한다.

    Args:
        input_dir: 재귀 탐색할 디렉토리
        inputs: 추가로 포함할 파일 목록

    Returns:
        중복 제거된 정렬된 파일 경로 목록
    """
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
    """파일에서 텍스트 스트림을 생성한다.

    .txt, .jsonl, .json 형식을 지원하며, 빈 줄은 자동으로 건너뛴다.

    Args:
        files: 읽을 파일 경로 목록
        text_key: JSON 객체에서 텍스트를 추출할 키 이름
        encoding: 파일 인코딩

    Yields:
        텍스트 문자열 (빈 줄 제외)
    """
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
    """토큰 빈도를 parquet로 저장한다.

    빈도 내림차순, 토큰 ID 오름차순으로 정렬하여 저장한다.

    Args:
        counter: 토큰 ID별 빈도 카운터
        output_path: Parquet 파일 저장 경로
    """
    rows = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    token_ids = [token_id for token_id, _ in rows]
    frequencies = [frequency for _, frequency in rows]
    table = pa.Table.from_pydict({"token_id": token_ids, "frequency": frequencies})
    pq.write_table(table, output_path)


def iter_encoded_chunks(
    texts: Iterable[str],
    tokenizer: GPT2Tokenizer,
    workers: int,
    chunk_size: int,
) -> Iterator[list[list[int]]]:
    """텍스트를 토큰화하여 청크 단위로 반환한다.

    ThreadPoolExecutor를 사용하여 병렬 토큰화를 수행한다.
    tokenizer 호출은 thread-safe하지 않을 수 있으므로 Lock으로 보호한다.

    Args:
        texts: 토큰화할 텍스트 iterable
        tokenizer: GPT-2 토크나이저
        workers: 스레드 워커 수
        chunk_size: 청크당 텍스트 개수

    Yields:
        토큰 ID 시퀀스 리스트의 리스트
    """
    counter: Counter[int] = Counter()
    encode_lock = Lock()

    def chunk_iter(source: Iterable[str]) -> Iterator[list[str]]:
        iterator = iter(source)
        while True:
            chunk = list(islice(iterator, chunk_size))
            if not chunk:
                break
            yield chunk

    def encode_chunk(chunk: list[str]) -> list[list[int]]:
        with encode_lock:
            output: BatchEncoding = tokenizer(
                chunk,
                add_special_tokens=False,
                truncation=True,
                max_length=tokenizer.model_max_length,
            )
        return cast(list[list[int]], output["input_ids"])

    with ThreadPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(encode_chunk, chunk_iter(texts))


def analyze_token_frequency(
    input_dir: Path,
    inputs: list[Path],
    output_frequency: Path,
    tokenizer_dir: Path,
    text_key: str,
    workers: int,
    chunk_size: int,
    max_texts: int,
    encoding: str,
) -> tuple[Iterator[list[list[int]]], GPT2Tokenizer]:
    """토큰 빈도를 분석한다.

    Args:
        input_dir: 코퍼스 입력 디렉토리
        inputs: 개별 입력 파일 목록
        output_sequences: BPE 토큰 시퀀스 출력 경로
        output_frequency: 토큰 빈도 parquet 출력 경로
        tokenizer_dir: 원본 토크나이저 디렉토리
        text_key: json/jsonl 텍스트 키
        workers: 스레드 워커 수 (0이면 CPU - 1)
        chunk_size: 스레드 청크 크기 (0이면 자동 설정)
        max_texts: 처리할 최대 텍스트 수 (0이면 전체)
        encoding: 입력 파일 인코딩

    Returns:
        분석 결과 정보를 담은 딕셔너리

    Raises:
        FileNotFoundError: 원본 토크나이저 파일이 없는 경우
        SystemExit: 입력 파일을 찾을 수 없는 경우
    """
    # 1) 입력 파일 수집
    input_files = find_input_files(input_dir, inputs)
    if not input_files:
        raise SystemExit("입력 파일을 찾을 수 없습니다.")
    logger.info("입력 파일 %d개 탐색 완료", len(input_files))

    # 2) 텍스트 스트림 구성
    texts = iter_texts(input_files, text_key, encoding)
    if max_texts > 0:
        texts = islice(texts, max_texts)
        logger.info("최대 %d개 텍스트만 처리", max_texts)

    # 3) 토크나이저 로드
    tokenizer_files = list(tokenizer_dir.glob("*")) if tokenizer_dir.exists() else []
    has_tokenizer_files = any(
        f.name in ["tokenizer.json", "vocab.json", "merges.txt"]
        for f in tokenizer_files
    )
    if not has_tokenizer_files:
        raise FileNotFoundError(f"원본 토크나이저 파일이 없습니다: {tokenizer_dir}")

    logger.info("토크나이저 로드: %s", tokenizer_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(str(tokenizer_dir))

    # 4) 병렬 처리 파라미터 계산
    if workers <= 0:
        workers = max(1, (os.cpu_count() or 1) - 1)
    if chunk_size <= 0:
        chunk_size = workers * 32
    logger.info("토큰화 설정: workers=%d, chunk_size=%d", workers, chunk_size)

    # 5) 토큰화 이터레이터 생성 (빈도 집계는 호출자에서 수행)
    encoded_chunks_iterator = iter_encoded_chunks(
        texts,
        tokenizer=tokenizer,
        workers=workers,
        chunk_size=chunk_size,
    )

    return encoded_chunks_iterator, tokenizer
