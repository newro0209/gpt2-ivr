"""기존 모델 임베딩 추출 로직"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from gpt2_ivr.utils.logging_config import get_logger


def extract_embeddings(
    model_name_or_path: str = "openai-community/gpt2",
    output_path: Path = Path("artifacts/embeddings/original_embeddings.pt"),
) -> dict[str, torch.Tensor]:
    """사전학습된 모델에서 임베딩 추출
    
    Args:
        model_name_or_path: 사전학습된 모델 이름 또는 경로
        output_path: 임베딩 저장 경로
        
    Returns:
        추출된 임베딩 딕셔너리
    """
    logger = get_logger("gpt2_ivr.embedding.extract")
    
    logger.info(f"🤖 모델 로드: {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    
    # Input embeddings (token embeddings) 추출
    input_embeddings = model.get_input_embeddings()
    wte = input_embeddings.weight.data.clone()
    
    # Output embeddings (LM head) 추출 (GPT-2는 tied weights 사용)
    output_embeddings = model.get_output_embeddings()
    lm_head = output_embeddings.weight.data.clone()
    
    logger.info(f"📊 Input embeddings shape: {wte.shape}")
    logger.info(f"📊 Output embeddings shape: {lm_head.shape}")
    
    embeddings = {
        "wte": wte,  # Word Token Embeddings
        "lm_head": lm_head,  # Language Model Head
    }
    
    # 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path)
    logger.info(f"💾 임베딩 저장: {output_path}")
    
    return embeddings
