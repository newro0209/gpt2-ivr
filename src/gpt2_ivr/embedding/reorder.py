"""remap 규칙 기준 임베딩 재정렬 로직"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from tokenizers import Tokenizer

from gpt2_ivr.utils.logging_config import get_logger


def reorder_embeddings(
    original_embeddings_path: Path = Path("artifacts/embeddings/original_embeddings.pt"),
    original_tokenizer_path: Path = Path("artifacts/tokenizers/original/tokenizer.json"),
    remapped_tokenizer_path: Path = Path("artifacts/tokenizers/remapped/tokenizer.json"),
    remap_rules_path: Path = Path("src/gpt2_ivr/tokenizer/remap_rules.yaml"),
    output_path: Path = Path("artifacts/embeddings/reordered_embeddings.pt"),
) -> dict[str, torch.Tensor]:
    """토큰 재할당 규칙에 따라 임베딩 재정렬
    
    Args:
        original_embeddings_path: 원본 임베딩 파일 경로
        original_tokenizer_path: 원본 토크나이저 경로
        remapped_tokenizer_path: 재할당된 토크나이저 경로
        remap_rules_path: 재할당 규칙 파일 경로
        output_path: 재정렬된 임베딩 저장 경로
        
    Returns:
        재정렬된 임베딩 딕셔너리
    """
    logger = get_logger("gpt2_ivr.embedding.reorder")
    
    # 원본 임베딩 로드
    logger.info(f"📂 원본 임베딩 로드: {original_embeddings_path}")
    embeddings = torch.load(original_embeddings_path)
    wte = embeddings["wte"]
    lm_head = embeddings["lm_head"]
    
    logger.info(f"📊 원본 임베딩 shape: {wte.shape}")
    
    # 토크나이저 로드
    logger.info(f"📖 원본 토크나이저 로드: {original_tokenizer_path}")
    original_tokenizer = Tokenizer.from_file(str(original_tokenizer_path))
    
    logger.info(f"📖 재할당 토크나이저 로드: {remapped_tokenizer_path}")
    remapped_tokenizer = Tokenizer.from_file(str(remapped_tokenizer_path))
    
    # 재할당 규칙 로드
    if remap_rules_path.exists():
        with open(remap_rules_path, "r", encoding="utf-8") as f:
            remap_rules = yaml.safe_load(f)
            if remap_rules is None:
                remap_rules = {}
    else:
        logger.warning(f"⚠️ 재할당 규칙 파일 없음: {remap_rules_path}")
        remap_rules = {}
    
    logger.info(f"📋 재할당 규칙 개수: {len(remap_rules)}")
    
    # 새 vocab size
    new_vocab_size = remapped_tokenizer.get_vocab_size()
    embedding_dim = wte.shape[1]
    
    # 새 임베딩 텐서 초기화
    new_wte = torch.zeros(new_vocab_size, embedding_dim, dtype=wte.dtype)
    new_lm_head = torch.zeros(new_vocab_size, embedding_dim, dtype=lm_head.dtype)
    
    logger.info(f"📊 새 임베딩 shape: {new_wte.shape}")
    
    # 기존 토큰에 대한 임베딩 복사
    original_vocab = original_tokenizer.get_vocab()
    remapped_vocab = remapped_tokenizer.get_vocab()
    
    copied_count = 0
    new_count = 0
    remapped_count = 0
    
    for token, new_id in remapped_vocab.items():
        if token in original_vocab:
            # 기존 토큰: 원본 임베딩 복사
            old_id = original_vocab[token]
            if old_id < wte.shape[0]:
                new_wte[new_id] = wte[old_id]
                new_lm_head[new_id] = lm_head[old_id]
                copied_count += 1
        else:
            # 새 토큰: remap 규칙 확인
            is_remapped = False
            for old_token, new_token in remap_rules.items():
                if new_token == token and old_token in original_vocab:
                    # 재할당된 토큰: 원본 토큰의 임베딩 복사
                    old_id = original_vocab[old_token]
                    if old_id < wte.shape[0]:
                        new_wte[new_id] = wte[old_id]
                        new_lm_head[new_id] = lm_head[old_id]
                        remapped_count += 1
                        is_remapped = True
                        logger.debug(f"  재할당: '{old_token}' (id:{old_id}) -> '{new_token}' (id:{new_id})")
                        break
            
            if not is_remapped:
                # 완전히 새로운 토큰: 평균값으로 초기화
                new_wte[new_id] = wte.mean(dim=0)
                new_lm_head[new_id] = lm_head.mean(dim=0)
                new_count += 1
    
    logger.info(f"✅ 임베딩 재정렬 완료:")
    logger.info(f"  - 복사된 토큰: {copied_count}개")
    logger.info(f"  - 재할당된 토큰: {remapped_count}개")
    logger.info(f"  - 새로 초기화된 토큰: {new_count}개")
    
    reordered_embeddings = {
        "wte": new_wte,
        "lm_head": new_lm_head,
    }
    
    # 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(reordered_embeddings, output_path)
    logger.info(f"💾 재정렬된 임베딩 저장: {output_path}")
    
    return reordered_embeddings
