"""원본 코퍼스를 정제하여 일관된 텍스트 형식으로 변환하는 헬퍼입니다."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, Sequence

logger = logging.getLogger(__name__)

_ALLOWED_CORPUS_SUFFIXES = {".txt", ".jsonl", ".json"}


def _collect_raw_files(raw_dir: Path) -> Sequence[Path]:
    """원본 코퍼스 디렉토리에서 허용된 확장자를 가진 파일 목록을 가져옵니다."""
    if not raw_dir.exists():
        return []

    files: list[Path] = []
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in _ALLOWED_CORPUS_SUFFIXES:
            files.append(path)
    return files


def _extract_texts_from_file(path: Path, text_key: str, encoding: str) -> Iterator[str]:
    """파일로부터 텍스트를 순차적으로 추출합니다."""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        with path.open("r", encoding=encoding) as handle:
            for line in handle:
                text = line.rstrip("\n").strip()
                if text:
                    yield text
        return

    if suffix == ".jsonl":
        with path.open("r", encoding=encoding) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "%s 파일의 jsonl 라인을 해석할 수 없습니다: %s", path, exc
                    )
                    continue
                if isinstance(record, dict) and text_key in record:
                    value = record[text_key]
                    if isinstance(value, str):
                        text = value.strip()
                        if text:
                            yield text
        return

    if suffix == ".json":
        try:
            with path.open("r", encoding=encoding) as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            logger.warning("%s 파일을 json으로 파싱할 수 없습니다: %s", path, exc)
            return
        if isinstance(payload, list):
            for record in payload:
                if isinstance(record, dict) and text_key in record:
                    text = str(record[text_key]).strip()
                    if text:
                        yield text
        elif isinstance(payload, dict):
            if text_key in payload:
                text = str(payload[text_key]).strip()
                if text:
                    yield text
        return


def normalize_raw_corpora(
    raw_dir: Path,
    cleaned_dir: Path,
    *,
    text_key: str,
    encoding: str,
    force: bool,
) -> list[Path]:
    """raw 디렉토리의 파일을 정제된 텍스트(.txt)로 변환합니다."""
    raw_files = _collect_raw_files(raw_dir)
    if not raw_files:
        logger.warning("raw 코퍼스 파일을 찾을 수 없습니다: %s", raw_dir)
        return []

    cleaned_dir.mkdir(parents=True, exist_ok=True)
    normalized_paths: list[Path] = []

    for raw_path in raw_files:
        relative = raw_path.relative_to(raw_dir)
        cleaned_path = (cleaned_dir / relative).with_suffix(".txt")
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)

        if cleaned_path.exists() and not force:
            raw_mtime = raw_path.stat().st_mtime
            cleaned_mtime = cleaned_path.stat().st_mtime
            if raw_mtime <= cleaned_mtime:
                logger.debug("정제본이 최신입니다: %s", cleaned_path)
                continue

        line_count = 0
        with cleaned_path.open("w", encoding="utf-8") as output_handle:
            for text in _extract_texts_from_file(raw_path, text_key, encoding):
                output_handle.write(text)
                output_handle.write("\n")
                line_count += 1

        if line_count == 0:
            cleaned_path.unlink(missing_ok=True)
            logger.warning(
                "%s에서 텍스트를 추출하지 못했습니다. 정제본을 제거합니다.", raw_path
            )
            continue

        normalized_paths.append(cleaned_path)
        logger.info("%s → %s (%d줄) 변환 완료", raw_path, cleaned_path, line_count)

    if normalized_paths:
        logger.info(
            "총 %d개의 코퍼스를 정제하여 '%s'에 저장했습니다.",
            len(normalized_paths),
            cleaned_dir,
        )
    else:
        logger.warning("변환된 정제본이 없습니다.")

    return normalized_paths
