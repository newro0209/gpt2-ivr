"""BPE 토큰 시퀀스 생성 모듈.

코퍼스 파일에서 텍스트를 읽어 GPT-2 BPE 토크나이저로 토큰화하여
토큰 ID 시퀀스를 생성한다. 병렬 처리를 통해 대용량 코퍼스를 효율적으로 처리한다.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from threading import Lock
from typing import Iterable, Iterator, cast

from transformers import BatchEncoding, GPT2Tokenizer

logger = logging.getLogger(__name__)


def find_input_files(input_dir: Path, inputs: list[Path]) -> list[Path]:
    """분석 대상 파일 목록을 수집한다.

    input_dir에서 재귀적으로 파일을 탐색하고, inputs 리스트의 파일을 추가한다.
    .txt 확장자만 허용한다.

    Args:
        input_dir: 재귀 탐색할 디렉토리
        inputs: 추가로 포함할 파일 목록

    Returns:
        중복 제거된 정렬된 파일 경로 목록
    """
    allowed_suffixes = {".txt"}
    files: list[Path] = []

    if input_dir.exists():
        for path in sorted(input_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in allowed_suffixes:
                files.append(path)

    for path in inputs:
        if path.is_file():
            files.append(path)

    return sorted(set(files))


def chunk_iter(source: Iterable[str], chunk_size: int) -> Iterator[list[str]]:
    """텍스트 스트림을 고정 크기 청크로 분할한다.

    Args:
        source: 텍스트 iterable
        chunk_size: 청크당 텍스트 개수

    Yields:
        chunk_size 크기의 텍스트 리스트
    """
    iterator = iter(source)
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            break
        yield chunk


def iter_bpe_token_sequences(
    texts: Iterable[str],
    tokenizer: GPT2Tokenizer,
    workers: int,
    chunk_size: int,
) -> Iterator[list[list[int]]]:
    """텍스트를 GPT-2 BPE token id 시퀀스로 변환한다.

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
    encode_lock = Lock()

    def encode_chunk(chunk: list[str]) -> list[list[int]]:
        with encode_lock:
            encoded: BatchEncoding = tokenizer(
                chunk,
                add_special_tokens=False,
                truncation=True,
                max_length=tokenizer.model_max_length,
            )
        return cast(list[list[int]], encoded["input_ids"])

    with ThreadPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(encode_chunk, chunk_iter(texts, chunk_size))
