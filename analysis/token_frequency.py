from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import os
from pathlib import Path
from threading import Lock
from transformers import BatchEncoding, GPT2Tokenizer
from typing import Iterable, Iterator, cast

from tqdm import tqdm


def build_logger() -> logging.Logger:
    """로거를 설정한다."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    return logging.getLogger("token-frequency")


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
    import pyarrow as pa
    import pyarrow.parquet as pq

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
                unit="문장",
            ):
                # 청크 결과 누적 및 기록
                for token_ids in chunk_ids:
                    counter.update(token_ids)
                    handle.write(" ".join(str(token_id) for token_id in token_ids))
                    handle.write("\n")

    return counter


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="GPT-2 BPE 토큰 빈도 분석")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("corpora/cleaned"),
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
        "--output-sequences",
        type=Path,
        default=Path("analysis/reports/bpe_token_id_sequences.txt"),
        help="BPE 토큰 id 시퀀스 출력 경로",
    )
    parser.add_argument(
        "--output-frequency",
        type=Path,
        default=Path("analysis/reports/token_frequency.parquet"),
        help="토큰 빈도 parquet 출력 경로",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai-community/gpt2",
        help="GPT-2 모델 이름",
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
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="입력 파일 인코딩",
    )
    return parser.parse_args()


def main() -> None:
    """엔트리 포인트."""
    # CLI 설정 로드
    args = parse_args()
    # 로깅 초기화
    logger = build_logger()

    # 입력 파일 목록 수집
    input_files = find_input_files(args.input_dir, args.input)
    if not input_files:
        raise SystemExit("입력 파일을 찾을 수 없습니다.")

    logger.info("입력 파일 %d개를 탐색했습니다.", len(input_files))

    # 텍스트 스트림 구성
    texts = iter_texts(input_files, args.text_key, args.encoding)
    if args.max_texts > 0:
        texts = islice(texts, args.max_texts)

    # 토크나이저 로드
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    # 토큰화 및 빈도 집계
    counter = collect_statistics(
        texts,
        output_sequences=args.output_sequences,
        tokenizer=tokenizer,
        workers=max(args.workers, 1),
        chunk_size=max(args.chunk_size, 1),
    )

    # 결과물 저장
    args.output_frequency.parent.mkdir(parents=True, exist_ok=True)
    write_frequency_parquet(counter, args.output_frequency)

    logger.info("토큰 빈도 parquet 저장 완료: %s", args.output_frequency)
    logger.info("토큰 시퀀스 저장 완료: %s", args.output_sequences)


if __name__ == "__main__":
    main()
