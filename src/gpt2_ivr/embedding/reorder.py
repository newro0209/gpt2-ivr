"""remap 규칙 기준 임베딩 재정렬 로직"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import yaml
from tokenizers import Tokenizer

from gpt2_ivr.utils.logging_config import get_logger


def reorder_embeddings(
    original_wte_path: Path,
    original_tokenizer_dir: Path,
    remapped_tokenizer_dir: Path,
    remap_rules_path: Path,
    output_dir: Path,
    logger: logging.Logger | None = None,
) -> dict[str, Path]:
    """
    Remap 규칙에 따라 토큰 임베딩을 재정렬한다.

    Args:
        original_wte_path: 원본 토큰 임베딩 파일 경로
        original_tokenizer_dir: 원본 토크나이저 디렉토리
        remapped_tokenizer_dir: 재할당된 토크나이저 디렉토리
        remap_rules_path: Remap 규칙 YAML 파일 경로
        output_dir: 출력 디렉토리
        logger: 로거 인스턴스 (선택사항)

    Returns:
        저장된 파일 경로를 담은 딕셔너리
    """
    if logger is None:
        logger = get_logger("gpt2_ivr.embedding.reorder")

    logger.info("🔄 임베딩 재정렬 시작")

    # 1. 원본 임베딩 로드
    logger.info("📥 원본 임베딩 로딩: %s", original_wte_path)
    original_wte = torch.load(original_wte_path)
    vocab_size, embedding_dim = original_wte.shape
    logger.info("원본 임베딩 shape: (%d, %d)", vocab_size, embedding_dim)

    # 2. 토크나이저 로드
    logger.info("📥 원본 토크나이저 로딩: %s", original_tokenizer_dir)
    original_tokenizer = Tokenizer.from_file(
        str(original_tokenizer_dir / "tokenizer.json")
    )

    logger.info("📥 Remapped 토크나이저 로딩: %s", remapped_tokenizer_dir)
    remapped_tokenizer = Tokenizer.from_file(
        str(remapped_tokenizer_dir / "tokenizer.json")
    )

    # 3. Remap 규칙 로드
    logger.info("📥 Remap 규칙 로딩: %s", remap_rules_path)
    with open(remap_rules_path, "r", encoding="utf-8") as f:
        remap_rules = yaml.safe_load(f) or {}
    logger.info("총 %d개의 remap 규칙 발견", len(remap_rules))

    # 4. 새로운 임베딩 텐서 생성
    new_vocab_size = remapped_tokenizer.get_vocab_size()
    logger.info("새로운 vocab 크기: %d", new_vocab_size)

    # Vocab 크기 검증
    if new_vocab_size < vocab_size:
        raise ValueError(
            f"새 vocab 크기({new_vocab_size})가 원본({vocab_size})보다 작습니다."
        )

    # 새 임베딩을 0으로 초기화
    aligned_wte = torch.zeros(new_vocab_size, embedding_dim, dtype=original_wte.dtype)

    if new_vocab_size == vocab_size:
        logger.info("✅ Vocab 크기 동일: %d", vocab_size)
    else:
        logger.info("⚠️ Vocab 크기 증가: %d -> %d", vocab_size, new_vocab_size)

    # 5. 먼저 기존 토큰들의 임베딩을 복사 (remap되지 않은 토큰 보존)
    original_vocab = original_tokenizer.get_vocab()
    remapped_vocab = remapped_tokenizer.get_vocab()
    preserved_count = 0

    for token, old_id in original_vocab.items():
        new_id = remapped_vocab.get(token)
        if new_id is not None and token not in remap_rules.values():
            # remap의 target이 아닌 토큰은 그대로 보존
            aligned_wte[new_id] = original_wte[old_id].clone()
            preserved_count += 1

    logger.info("✅ %d개의 기존 토큰 임베딩 보존", preserved_count)

    # 6. Remap 규칙에 따라 임베딩 재배치
    remap_count = 0
    for old_token, new_token in remap_rules.items():
        old_id = original_tokenizer.token_to_id(old_token)
        new_id = remapped_tokenizer.token_to_id(new_token)

        if old_id is not None and new_id is not None:
            # 원본 토큰의 임베딩을 새 토큰 위치로 복사
            aligned_wte[new_id] = original_wte[old_id].clone()
            remap_count += 1
            logger.debug(
                "재할당: '%s' (id:%d) -> '%s' (id:%d)",
                old_token,
                old_id,
                new_token,
                new_id,
            )

    logger.info("✅ 총 %d개의 임베딩 재할당 완료", remap_count)

    # 7. 출력 디렉토리 생성 및 저장
    output_dir.mkdir(parents=True, exist_ok=True)

    aligned_wte_path = output_dir / "aligned_wte.pt"
    torch.save(aligned_wte, aligned_wte_path)
    logger.info("💾 재정렬된 임베딩 저장: %s", aligned_wte_path)

    # 8. 메타데이터 저장
    metadata = {
        "original_vocab_size": vocab_size,
        "new_vocab_size": new_vocab_size,
        "embedding_dim": embedding_dim,
        "preserved_count": preserved_count,
        "remap_count": remap_count,
        "aligned_wte_shape": list(aligned_wte.shape),
    }

    metadata_path = output_dir / "reorder_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("📋 메타데이터 저장: %s", metadata_path)

    return {
        "aligned_wte": aligned_wte_path,
        "metadata": metadata_path,
    }
