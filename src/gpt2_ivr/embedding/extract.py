"""기존 모델 임베딩 추출 로직"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel

from gpt2_ivr.utils.logging_config import get_logger


def extract_embeddings(
    model_name: str,
    output_dir: Path,
    logger: logging.Logger | None = None,
) -> dict[str, Path]:
    """
    GPT-2 모델에서 토큰 임베딩(wte)과 위치 임베딩(wpe)을 추출하여 저장한다.

    Args:
        model_name: Hugging Face Hub 모델 이름
        output_dir: 임베딩 저장 디렉토리
        logger: 로거 인스턴스 (선택사항)

    Returns:
        저장된 임베딩 파일 경로를 담은 딕셔너리
    """
    if logger is None:
        logger = get_logger("gpt2_ivr.embedding.extract")

    logger.info("🔍 모델 로딩 중: %s", model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # 임베딩 추출
    wte = model.transformer.wte.weight.data.clone()  # Token embeddings
    wpe = model.transformer.wpe.weight.data.clone()  # Position embeddings

    logger.info(
        "✅ 임베딩 추출 완료 - wte shape: %s, wpe shape: %s",
        wte.shape,
        wpe.shape,
    )

    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    # 임베딩 저장
    wte_path = output_dir / "original_wte.pt"
    wpe_path = output_dir / "original_wpe.pt"

    torch.save(wte, wte_path)
    torch.save(wpe, wpe_path)

    logger.info("💾 Token embedding 저장: %s", wte_path)
    logger.info("💾 Position embedding 저장: %s", wpe_path)

    # 메타데이터 저장
    metadata = {
        "model_name": model_name,
        "vocab_size": wte.shape[0],
        "embedding_dim": wte.shape[1],
        "max_position_embeddings": wpe.shape[0],
        "wte_shape": list(wte.shape),
        "wpe_shape": list(wpe.shape),
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("📋 메타데이터 저장: %s", metadata_path)

    return {
        "wte": wte_path,
        "wpe": wpe_path,
        "metadata": metadata_path,
    }
