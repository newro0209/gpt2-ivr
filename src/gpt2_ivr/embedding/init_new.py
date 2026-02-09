"""신규 토큰 임베딩 초기화 로직"""

from __future__ import annotations

from pathlib import Path

import torch

from gpt2_ivr.utils.logging_config import get_logger


def initialize_new_embeddings(
    reordered_embeddings_path: Path = Path("artifacts/embeddings/reordered_embeddings.pt"),
    model_save_path: Path = Path("artifacts/embeddings/initialized_model"),
    model_name_or_path: str = "openai-community/gpt2",
) -> dict[str, str]:
    """재정렬된 임베딩을 모델에 적용하고 저장
    
    Args:
        reordered_embeddings_path: 재정렬된 임베딩 파일 경로
        model_save_path: 초기화된 모델 저장 경로
        model_name_or_path: 기본 모델 이름 또는 경로
        
    Returns:
        결과 딕셔너리
    """
    logger = get_logger("gpt2_ivr.embedding.init_new")
    
    from transformers import AutoConfig, AutoModelForCausalLM
    
    # 재정렬된 임베딩 로드
    logger.info(f"📂 재정렬된 임베딩 로드: {reordered_embeddings_path}")
    embeddings = torch.load(reordered_embeddings_path)
    new_wte = embeddings["wte"]
    new_lm_head = embeddings["lm_head"]
    
    new_vocab_size = new_wte.shape[0]
    logger.info(f"📊 새 vocab size: {new_vocab_size}")
    
    # 모델 로드
    logger.info(f"🤖 모델 로드: {model_name_or_path}")
    config = AutoConfig.from_pretrained(model_name_or_path)
    config.vocab_size = new_vocab_size
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True,
    )
    
    # 임베딩 레이어 크기 조정
    model.resize_token_embeddings(new_vocab_size)
    
    # 재정렬된 임베딩 적용
    logger.info("🔧 재정렬된 임베딩을 모델에 적용")
    with torch.no_grad():
        model.get_input_embeddings().weight.copy_(new_wte)
        model.get_output_embeddings().weight.copy_(new_lm_head)
    
    # 모델 저장
    model_save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(model_save_path))
    logger.info(f"💾 초기화된 모델 저장: {model_save_path}")
    
    logger.info("✅ 신규 토큰 임베딩 초기화 완료")
    
    return {
        "status": "success",
        "model_path": str(model_save_path),
        "vocab_size": new_vocab_size,
    }
