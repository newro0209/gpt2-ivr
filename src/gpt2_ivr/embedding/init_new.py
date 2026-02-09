"""신규 토큰 임베딩 초기화 로직.

재할당 과정에서 새로 추가된 토큰의 임베딩을 초기화한다.
세 가지 전략을 지원한다:
- mean: 기존 토큰 임베딩의 평균으로 초기화
- random: 기존 토큰 임베딩의 표준편차를 사용한 정규분포에서 샘플링
- zeros: 0으로 유지
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import yaml
from tokenizers import Tokenizer


def initialize_new_token_embeddings(
    aligned_wte_path: Path,
    original_tokenizer_dir: Path,
    remapped_tokenizer_dir: Path,
    remap_rules_path: Path,
    output_dir: Path,
    init_strategy: str = "mean",
    logger: logging.Logger | None = None,
) -> dict[str, Path]:
    """신규 추가된 토큰에 대한 임베딩을 초기화한다.

    Args:
        aligned_wte_path: 재정렬된 토큰 임베딩 파일 경로
        original_tokenizer_dir: 원본 토크나이저 디렉토리
        remapped_tokenizer_dir: 재할당된 토크나이저 디렉토리
        remap_rules_path: Remap 규칙 YAML 파일 경로
        output_dir: 출력 디렉토리
        init_strategy: 초기화 전략 ('mean', 'random', 'zeros')
        logger: 로거 인스턴스 (선택사항)

    Returns:
        저장된 파일 경로를 담은 딕셔너리
    """
    if logger is None:
        logger = logging.getLogger("gpt2_ivr.embedding.init_new")

    logger.info("신규 토큰 임베딩 초기화 시작")
    logger.info("초기화 전략: %s", init_strategy)

    # 1. 재정렬된 임베딩 로드
    logger.info("재정렬된 임베딩 로드: %s", aligned_wte_path)
    aligned_wte = torch.load(aligned_wte_path)
    vocab_size, embedding_dim = aligned_wte.shape
    logger.info("임베딩 shape: (%d, %d)", vocab_size, embedding_dim)

    # 2. 토크나이저 로드
    logger.info("원본 토크나이저 로드: %s", original_tokenizer_dir)
    original_tokenizer = Tokenizer.from_file(
        str(original_tokenizer_dir / "tokenizer.json")
    )

    logger.info("재할당 토크나이저 로드: %s", remapped_tokenizer_dir)
    remapped_tokenizer = Tokenizer.from_file(
        str(remapped_tokenizer_dir / "tokenizer.json")
    )

    # 3. Remap 규칙 로드
    logger.info("재할당 규칙 로드: %s", remap_rules_path)
    with open(remap_rules_path, "r", encoding="utf-8") as f:
        remap_rules = yaml.safe_load(f) or {}

    # 4. 신규 토큰 탐색 (remapped vocab에 있지만 original vocab에는 없는 토큰)
    # 1) original에 있던 토큰은 제외
    # 2) remap target이지만 source가 original에 있던 경우도 제외 (재할당이므로 신규 아님)
    # 3) 임베딩이 0으로 초기화된 경우만 신규로 판단
    new_tokens = []
    remapped_vocab = remapped_tokenizer.get_vocab()

    for token, token_id in remapped_vocab.items():
        if original_tokenizer.token_to_id(token) is not None:
            continue

        is_remapped_target = token in remap_rules.values()
        if is_remapped_target:
            source_existed = any(
                original_tokenizer.token_to_id(old_token) is not None
                for old_token, new_token in remap_rules.items()
                if new_token == token
            )
            if source_existed:
                continue

        if token_id < vocab_size and torch.all(aligned_wte[token_id] == 0):
            new_tokens.append((token, token_id))

    logger.info("신규 토큰 %d개 발견", len(new_tokens))

    # 5. 초기화 전략에 따라 임베딩 초기화
    if len(new_tokens) > 0:
        non_zero_mask = ~torch.all(aligned_wte == 0, dim=1)
        has_non_zero = non_zero_mask.sum() > 0

        if init_strategy == "mean":
            if has_non_zero:
                mean_embedding = aligned_wte[non_zero_mask].mean(dim=0)
            else:
                mean_embedding = torch.zeros(embedding_dim)

            for token, token_id in new_tokens:
                aligned_wte[token_id] = mean_embedding
                logger.debug(
                    "토큰 '%s' (id:%d) -> mean 임베딩으로 초기화", token, token_id
                )

        elif init_strategy == "random":
            if has_non_zero:
                std = aligned_wte[non_zero_mask].std().item()
            else:
                std = 0.02

            for token, token_id in new_tokens:
                aligned_wte[token_id] = torch.randn(embedding_dim) * std
                logger.debug(
                    "토큰 '%s' (id:%d) -> random 임베딩으로 초기화 (std=%.4f)",
                    token,
                    token_id,
                    std,
                )

        elif init_strategy == "zeros":
            logger.info("zeros 전략 선택 - 기존 zero 임베딩 유지")

        else:
            raise ValueError(f"알 수 없는 초기화 전략: {init_strategy}")

        logger.info("신규 토큰 임베딩 %d개 초기화 완료", len(new_tokens))
    else:
        logger.info("초기화할 신규 토큰이 없습니다")

    # 6. 출력 디렉토리 생성 및 저장
    output_dir.mkdir(parents=True, exist_ok=True)

    final_wte_path = output_dir / "final_wte.pt"
    torch.save(aligned_wte, final_wte_path)
    logger.info("최종 임베딩 저장: %s", final_wte_path)

    # 7. 메타데이터 저장
    metadata = {
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "new_tokens_count": len(new_tokens),
        "init_strategy": init_strategy,
        "new_tokens": [
            {"token": token, "id": token_id} for token, token_id in new_tokens
        ],
    }

    metadata_path = output_dir / "init_new_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("메타데이터 저장: %s", metadata_path)

    return {
        "final_wte": final_wte_path,
        "metadata": metadata_path,
    }
