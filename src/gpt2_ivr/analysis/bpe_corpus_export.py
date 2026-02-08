from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import os
from pathlib import Path
from threading import Lock
from typing import Iterable, Iterator, cast

from tqdm import tqdm
from transformers import BatchEncoding, GPT2Tokenizer

from gpt2_ivr.utils.logging_config import get_logger

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
            for chunk_ids in tqdm(
                executor.map(encode_chunk, chunk_iter(texts, chunk_size)),
                desc="BPE 토큰 시퀀스 생성",
                unit="청크",
            ):
                for token_ids in chunk_ids:
                    handle.write(" ".join(str(token_id) for token_id in token_ids))
                    handle.write("\n")
                    total_lines += 1
                    total_tokens += len(token_ids)

    return total_lines, total_tokens


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(
        description="코퍼스를 GPT-2 BPE token id 시퀀스로 변환한다.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("artifacts/corpora/cleaned"),
        help="코퍼스 입력 디렉토리",
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=[],
        help="개별 입력 파일 경로",
    )
    parser.add_argument(
        "--text-key",
        type=str,
        default="text",
        help="json/jsonl 텍스트 키",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="입력 파일 인코딩",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai-community/gpt2",
        help="GPT-2 모델 이름",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/analysis/reports/bpe_token_id_sequences.txt"),
        help="BPE token id 시퀀스 출력 경로",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, int((os.cpu_count() or 1) * 2)),
        help="스레드 워커 수",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="스레드 청크 크기",
    )
    parser.add_argument(
        "--max-texts",
        type=int,
        default=0,
        help="처리할 최대 텍스트 수 (0이면 전체)",
    )
    return parser.parse_args()


def main() -> None:
    """엔트리 포인트."""
    args = parse_args()

    input_files = find_input_files(args.input_dir, args.input)
    if not input_files:
        raise SystemExit("입력 파일을 찾을 수 없습니다.")

    logger.info("입력 파일 %d개를 탐색했습니다.", len(input_files))

    texts: Iterable[str] = iter_texts(input_files, args.text_key, args.encoding)
    if args.max_texts > 0:
        texts = islice(texts, args.max_texts)
        logger.info("최대 텍스트 수 제한을 적용합니다: %d", args.max_texts)

    logger.info("토크나이저를 로드합니다: %s", args.model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)

    line_count, token_count = export_bpe_token_sequences(
        texts=texts,
        output_path=args.output,
        tokenizer=tokenizer,
        workers=max(args.workers, 1),
        chunk_size=max(args.chunk_size, 1),
    )

    logger.info("BPE token id 시퀀스 저장 완료: %s", args.output)
    logger.info("생성된 시퀀스 수: %d", line_count)
    logger.info("누적 토큰 수: %d", token_count)


if __name__ == "__main__":
    main()
