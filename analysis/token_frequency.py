from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - 선택적 의존성
    tqdm = None


_TOKENIZER = None


def build_logger() -> logging.Logger:
    """로거를 설정한다."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    return logging.getLogger("token-frequency")


def load_tokenizer(model_name: str):
    """GPT-2 BPE 토크나이저를 로드한다."""
    try:
        from transformers import GPT2TokenizerFast
    except ImportError as exc:  # pragma: no cover - 런타임 의존성
        raise SystemExit(
            "transformers 패키지가 필요합니다. `uv pip install transformers`로 설치하세요."
        ) from exc

    return GPT2TokenizerFast.from_pretrained(model_name)


def init_worker(model_name: str) -> None:
    """멀티프로세스 워커 초기화를 수행한다."""
    global _TOKENIZER
    _TOKENIZER = load_tokenizer(model_name)


def encode_in_worker(text: str) -> list[int]:
    """워커에서 텍스트를 토큰 id 시퀀스로 변환한다."""
    if _TOKENIZER is None:
        raise RuntimeError("토크나이저가 초기화되지 않았습니다.")
    return _TOKENIZER.encode(text, add_special_tokens=False)


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
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - 런타임 의존성
        raise SystemExit(
            "pyarrow 패키지가 필요합니다. `uv pip install pyarrow`로 설치하세요."
        ) from exc

    rows = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    token_ids = [token_id for token_id, _ in rows]
    frequencies = [frequency for _, frequency in rows]
    table = pa.Table.from_pydict({"token_id": token_ids, "frequency": frequencies})
    pq.write_table(table, output_path)


def maybe_wrap_progress(iterable: Iterable[list[int]], logger: logging.Logger, log_every: int):
    """진행 표시를 위한 래퍼를 반환한다."""
    if tqdm is not None:
        return tqdm(iterable, desc="토큰화", unit="문장")

    def generator():
        for index, item in enumerate(iterable, start=1):
            if index % log_every == 0:
                logger.info("진행 상황: %d 문장 처리", index)
            yield item

    return generator()


def collect_statistics(
    texts: Iterable[str],
    output_sequences: Path,
    model_name: str,
    workers: int,
    chunk_size: int,
    log_every: int,
) -> Counter[int]:
    """토큰 시퀀스와 빈도 통계를 생성한다."""
    counter: Counter[int] = Counter()

    output_sequences.parent.mkdir(parents=True, exist_ok=True)
    with output_sequences.open("w", encoding="utf-8") as handle:
        if workers > 1:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=init_worker,
                initargs=(model_name,),
            ) as executor:
                for token_ids in maybe_wrap_progress(
                    executor.map(encode_in_worker, texts, chunksize=chunk_size),
                    logger=logging.getLogger("token-frequency"),
                    log_every=log_every,
                ):
                    counter.update(token_ids)
                    handle.write(" ".join(str(token_id) for token_id in token_ids))
                    handle.write("\n")
        else:
            tokenizer = load_tokenizer(model_name)
            for text in maybe_wrap_progress(
                (tokenizer.encode(item, add_special_tokens=False) for item in texts),
                logger=logging.getLogger("token-frequency"),
                log_every=log_every,
            ):
                counter.update(text)
                handle.write(" ".join(str(token_id) for token_id in text))
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
        default=1,
        help="프로세스 워커 수",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="멀티프로세싱 청크 크기",
    )
    parser.add_argument(
        "--max-texts",
        type=int,
        default=0,
        help="처리할 최대 텍스트 수 (0이면 전체)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="진행 로그 출력 간격",
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
    args = parse_args()
    logger = build_logger()

    input_files = find_input_files(args.input_dir, args.input)
    if not input_files:
        raise SystemExit("입력 파일을 찾을 수 없습니다.")

    logger.info("입력 파일 %d개를 탐색했습니다.", len(input_files))

    texts = iter_texts(input_files, args.text_key, args.encoding)
    if args.max_texts > 0:
        texts = islice(texts, args.max_texts)

    counter = collect_statistics(
        texts,
        output_sequences=args.output_sequences,
        model_name=args.model_name,
        workers=max(args.workers, 1),
        chunk_size=max(args.chunk_size, 1),
        log_every=max(args.log_every, 1),
    )

    args.output_frequency.parent.mkdir(parents=True, exist_ok=True)
    write_frequency_parquet(counter, args.output_frequency)

    logger.info("토큰 빈도 parquet 저장 완료: %s", args.output_frequency)
    logger.info("토큰 시퀀스 저장 완료: %s", args.output_sequences)


if __name__ == "__main__":
    main()
