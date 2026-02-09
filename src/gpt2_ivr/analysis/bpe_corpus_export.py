from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from threading import Lock
from typing import Iterable, Iterator, cast

from transformers import BatchEncoding, GPT2Tokenizer

from gpt2_ivr.utils.logging_config import create_progress, get_logger

logger = get_logger(__name__)


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
    """파일 목록에서 텍스트를 순차 생성한다."""
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


def chunk_iter(source: Iterable[str], chunk_size: int) -> Iterator[list[str]]:
    """텍스트 스트림을 고정 크기 청크로 분할한다."""
    iterator = iter(source)
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            break
        yield chunk


def export_bpe_token_sequences(
    texts: Iterable[str],
    output_path: Path,
    tokenizer: GPT2Tokenizer,
    workers: int,
    chunk_size: int,
) -> tuple[int, int]:
    """텍스트를 GPT-2 BPE token id 시퀀스로 변환해 파일로 저장한다."""
    encode_lock = Lock()
    total_lines = 0
    total_tokens = 0

    def encode_chunk(chunk: list[str]) -> list[list[int]]:
        # 토크나이저 인스턴스의 스레드 안전성을 보호한다.
        with encode_lock:
            encoded: BatchEncoding = tokenizer(
                chunk,
                add_special_tokens=False,
                truncation=True,
                max_length=tokenizer.model_max_length,
            )
        return cast(list[list[int]], encoded["input_ids"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            with create_progress() as progress:
                task_id = progress.add_task("BPE 토큰 시퀀스 생성", total=None)
                for chunk_ids in executor.map(
                    encode_chunk, chunk_iter(texts, chunk_size)
                ):
                    progress.advance(task_id)
                    for token_ids in chunk_ids:
                        handle.write(" ".join(str(token_id) for token_id in token_ids))
                        handle.write("\n")
                        total_lines += 1
                        total_tokens += len(token_ids)

    return total_lines, total_tokens
