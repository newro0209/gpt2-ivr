"""accelerate 기반 학습 실행 로직"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.tokenization_utils import PreTrainedTokenizer

from gpt2_ivr.utils.logging_config import get_logger


def load_training_config(config_path: Path) -> dict[str, Any]:
    """학습 설정 파일 로드
    
    Args:
        config_path: YAML 설정 파일 경로
        
    Returns:
        설정 딕셔너리
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_dataset(dataset_path: Path, tokenizer: PreTrainedTokenizer) -> dict[str, Any]:
    """데이터셋 로드 및 전처리
    
    Args:
        dataset_path: 데이터셋 디렉토리 경로
        tokenizer: 토크나이저 인스턴스
        
    Returns:
        토크나이징된 데이터셋
    """
    logger = get_logger("gpt2_ivr.training.load_dataset")
    
    # 코퍼스 파일 수집
    corpus_files = list(dataset_path.glob("*.txt"))
    if not corpus_files:
        raise FileNotFoundError(f"코퍼스 파일을 찾을 수 없습니다: {dataset_path}")
    
    logger.info(f"📂 {len(corpus_files)}개의 코퍼스 파일 발견")
    
    # 텍스트 로드
    texts = []
    for file_path in corpus_files:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                texts.append(text)
    
    logger.info(f"📝 {len(texts)}개의 텍스트 샘플 로드 완료")
    
    # 토크나이징
    tokenized_texts = tokenizer(
        texts,
        truncation=True,
        max_length=1024,
        padding=False,
        return_tensors=None,
    )
    
    logger.info(f"✅ 토크나이징 완료: {len(tokenized_texts['input_ids'])}개 시퀀스")
    
    return {
        "input_ids": tokenized_texts["input_ids"],
        "attention_mask": tokenized_texts["attention_mask"],
    }


def train_model(
    model_name_or_path: str = "openai-community/gpt2",
    tokenizer_path: Path = Path("artifacts/tokenizers/remapped"),
    dataset_path: Path = Path("artifacts/corpora/cleaned"),
    output_dir: Path = Path("artifacts/training/sft_checkpoint"),
    config_path: Path = Path("src/gpt2_ivr/training/sft_config.yaml"),
) -> dict[str, Any]:
    """모델 학습 실행
    
    Args:
        model_name_or_path: 사전학습된 모델 이름 또는 경로
        tokenizer_path: 토크나이저 경로 (remapped)
        dataset_path: 학습 데이터셋 경로
        output_dir: 체크포인트 저장 경로
        config_path: 학습 설정 파일 경로
        
    Returns:
        학습 결과 딕셔너리
    """
    logger = get_logger("gpt2_ivr.training")
    logger.info("🚀 학습 시작")
    
    # 설정 로드
    if config_path.exists():
        config = load_training_config(config_path)
        logger.info(f"⚙️ 설정 파일 로드: {config_path}")
    else:
        logger.warning(f"⚠️ 설정 파일 없음: {config_path}. 기본값 사용")
        config = {}
    
    # 토크나이저 로드
    tokenizer_json_path = tokenizer_path / "tokenizer.json"
    if not tokenizer_json_path.exists():
        raise FileNotFoundError(
            f"토크나이저 파일을 찾을 수 없습니다: {tokenizer_json_path}\n"
            f"먼저 'uv run ivr remap' 명령을 실행하세요."
        )
    
    logger.info(f"📖 토크나이저 로드: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        use_fast=True,
    )
    
    # 특수 토큰 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"🔧 pad_token을 eos_token으로 설정: {tokenizer.eos_token}")
    
    # 모델 로드
    logger.info(f"🤖 모델 로드: {model_name_or_path}")
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    
    # vocab size 업데이트 (IVR로 토큰이 추가된 경우)
    tokenizer_vocab_size = len(tokenizer)
    if model_config.vocab_size != tokenizer_vocab_size:
        logger.warning(
            f"⚠️ 모델 vocab size ({model_config.vocab_size})와 "
            f"토크나이저 vocab size ({tokenizer_vocab_size})가 다릅니다. "
            f"모델 vocab size를 {tokenizer_vocab_size}로 조정합니다."
        )
        model_config.vocab_size = tokenizer_vocab_size
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=model_config,
    )
    
    # 임베딩 크기 조정 (필요한 경우)
    if model.get_input_embeddings().weight.shape[0] != tokenizer_vocab_size:
        logger.info(
            f"🔧 임베딩 레이어 크기 조정: "
            f"{model.get_input_embeddings().weight.shape[0]} -> {tokenizer_vocab_size}"
        )
        model.resize_token_embeddings(tokenizer_vocab_size)
    
    # 데이터셋 로드
    logger.info(f"📚 데이터셋 로드: {dataset_path}")
    dataset = load_dataset(dataset_path, tokenizer)
    
    # Data collator 설정 (Language Modeling용)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM이므로 MLM 사용 안 함
    )
    
    # Training Arguments 설정
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=config.get("overwrite_output_dir", True),
        num_train_epochs=config.get("num_train_epochs", 3),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        learning_rate=config.get("learning_rate", 5e-5),
        weight_decay=config.get("weight_decay", 0.01),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.03),
        logging_dir=str(output_dir / "logs"),
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 500),
        save_total_limit=config.get("save_total_limit", 2),
        seed=config.get("seed", 42),
        report_to=config.get("report_to", "tensorboard"),
        remove_unused_columns=False,
    )
    
    logger.info(f"⚙️ TrainingArguments 설정 완료")
    logger.info(f"  - Epochs: {training_args.num_train_epochs}")
    logger.info(f"  - Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"  - Learning rate: {training_args.learning_rate}")
    
    # Trainer 생성
    # Dataset을 리스트 형태로 변환 (Trainer가 요구하는 형식)
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, encodings: dict[str, list[list[int]]]):
            self.encodings = encodings
        
        def __len__(self) -> int:
            return len(self.encodings["input_ids"])
        
        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            return {
                "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
                "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            }
    
    train_dataset = SimpleDataset(dataset)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # 학습 실행
    logger.info("🏃 학습 시작...")
    train_result = trainer.train()
    
    # 모델 저장
    logger.info(f"💾 모델 저장: {output_dir}")
    trainer.save_model(str(output_dir / "final_model"))
    tokenizer.save_pretrained(str(output_dir / "final_model"))
    
    # 학습 메트릭 저장
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    logger.info("✅ 학습 완료!")
    logger.info(f"📊 Loss: {metrics.get('train_loss', 'N/A')}")
    logger.info(f"⏱️ Runtime: {metrics.get('train_runtime', 'N/A')}s")
    
    return {
        "status": "success",
        "output_dir": str(output_dir),
        "final_model_dir": str(output_dir / "final_model"),
        "metrics": metrics,
    }
