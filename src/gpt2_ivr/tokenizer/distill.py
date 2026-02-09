"""토크나이저 증류 비즈니스 로직.

GPT-2 BPE 토크나이저를 Unigram 모델로 증류하여 원본과 유사한 동작을 하지만
확률 기반 토큰 분할이 가능한 토크나이저를 생성한다. 원본의 어휘 크기를 유지하면서
코퍼스를 기반으로 Unigram 모델을 학습한다.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterator, TypedDict, cast

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast

# 병렬 처리 활성화
os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = logging.getLogger(__name__)


class DistillResult(TypedDict):
    """증류 결과 타입.

    Attributes:
        output_dir: 증류된 토크나이저 저장 디렉토리
        vocab_size: 증류된 토크나이저의 어휘 크기
        original_vocab_size: 원본 토크나이저의 어휘 크기
    """

    output_dir: Path
    vocab_size: int
    original_vocab_size: int


def get_training_corpus(
    corpus_dir: Path, batch_size: int = 1000
) -> Iterator[list[str]]:
    """클린 코퍼스 디렉토리에서 텍스트 파일을 읽어 학습 코퍼스 이터레이터를 생성한다.

    Args:
        corpus_dir: 코퍼스 디렉토리 경로
        batch_size: 배치 크기

    Yields:
        텍스트 배치 리스트
    """
    files = list(corpus_dir.glob("*.txt"))
    if not files:
        logger.warning(
            "경고: %s 디렉토리에 텍스트 파일이 없습니다. 토크나이저 학습을 진행할 수 없습니다.",
            corpus_dir,
        )
        yield []
        return

    logger.info("코퍼스 파일 %d개 로드 중: %s", len(files), corpus_dir)
    batch: list[str] = []
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line:  # 빈 줄은 무시
                    batch.append(stripped_line)
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
    if batch:  # 남은 배치 처리
        yield batch


def distill_unigram_tokenizer(
    original_tokenizer_dir: Path,
    distilled_tokenizer_dir: Path,
    corpus_dir: Path,
) -> DistillResult:
    """GPT-2 BPE 토크나이저의 동작을 모방하는 Unigram 토크나이저를 distillation 방식으로 학습한다.

    어휘 크기는 원본 토크나이저와 동일하게 맞춘다.

    Args:
        original_tokenizer_dir: 원본 토크나이저 디렉토리
        distilled_tokenizer_dir: 증류된 토크나이저 저장 디렉토리
        corpus_dir: 학습 코퍼스 디렉토리
    Returns:
        증류 결과 정보를 담은 딕셔너리

    Raises:
        Exception: 토크나이저 로드 또는 학습 실패 시
    """
    logger.info("Unigram 토크나이저 증류 시작")

    # 1. GPT-2 BPE 토크나이저 로드
    tokenizer_files = (
        list(original_tokenizer_dir.glob("*"))
        if original_tokenizer_dir.exists()
        else []
    )

    has_tokenizer_files = any(
        f.name in ["tokenizer.json", "vocab.json", "merges.txt"]
        for f in tokenizer_files
    )

    if not has_tokenizer_files:
        raise FileNotFoundError(
            f"원본 토크나이저 파일이 없습니다: {original_tokenizer_dir}"
        )

    try:
        original_tokenizer = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(str(original_tokenizer_dir)),
        )
        logger.info("원본 토크나이저 로드 완료 (vocab_size: %d)", len(original_tokenizer.get_vocab()))
    except Exception as e:
        raise RuntimeError(
            f"원본 토크나이저 로드 실패: {original_tokenizer_dir}"
        ) from e

    original_vocab_size = len(original_tokenizer.get_vocab())
    vocab_size = original_vocab_size

    # 2. Unigram 토크나이저 초기화
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    logger.info("Unigram 모델 및 ByteLevel PreTokenizer 초기화 완료")

    # 3. 트레이너 설정
    special_tokens = original_tokenizer.all_special_tokens
    unk_token = original_tokenizer.unk_token or original_tokenizer.eos_token
    assert isinstance(unk_token, str)
    if unk_token not in special_tokens:
        special_tokens.append(unk_token)

    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size, special_tokens=special_tokens, unk_token=unk_token
    )
    logger.info("UnigramTrainer 설정 (vocab_size: %d, special_tokens: %s)", vocab_size, special_tokens)

    # 4. 코퍼스를 사용하여 토크나이저 학습
    logger.info("토크나이저 학습 시작: %s", corpus_dir)
    tokenizer.train_from_iterator(get_training_corpus(corpus_dir, batch_size=10000), trainer=trainer)
    logger.info("토크나이저 학습 완료")

    # 5. Distilled Unigram 토크나이저 저장
    distilled_tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_json_path = distilled_tokenizer_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json_path))
    logger.info("증류 토크나이저 저장: %s", tokenizer_json_path)

    # Hugging Face PreTrainedTokenizerFast와 호환되도록 추가 파일 생성
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=unk_token,
        model_max_length=original_tokenizer.model_max_length,
        bos_token=original_tokenizer.bos_token,
        eos_token=original_tokenizer.eos_token,
        pad_token=original_tokenizer.pad_token,
    )
    hf_tokenizer.save_pretrained(distilled_tokenizer_dir)
    logger.info("PreTrainedTokenizerFast 호환 파일 저장: %s", distilled_tokenizer_dir)

    return DistillResult(
        output_dir=distilled_tokenizer_dir,
        vocab_size=vocab_size,
        original_vocab_size=original_vocab_size,
    )
