"""Tokenizer Remapping Command"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tokenizers import Tokenizer

from gpt2_ivr.commands.base import Command
from gpt2_ivr.utils.logging_config import get_logger


class RemapCommand(Command):
    """Remap Command"""

    def __init__(
        self,
        distilled_tokenizer_dir: Path | str = "artifacts/tokenizers/distilled_unigram",
        remapped_tokenizer_dir: Path | str = "artifacts/tokenizers/remapped",
        remap_rules_path: Path | str = "src/gpt2_ivr/tokenizer/remap_rules.yaml",
        replacement_candidates_path: Path | str = "artifacts/analysis/reports/replacement_candidates.csv",
    ) -> None:
        """
        RemapCommand 초기화.

        Args:
            distilled_tokenizer_dir: 증류된 토크나이저 디렉토리 (상대 경로는 CWD 기준)
            remapped_tokenizer_dir: 재할당 토크나이저 디렉토리 (상대 경로는 CWD 기준)
            remap_rules_path: 재할당 규칙 파일 경로 (상대 경로는 CWD 기준)
            replacement_candidates_path: 교체 후보 CSV 경로 (상대 경로는 CWD 기준)

        Note:
            모든 경로는 현재 작업 디렉토리(CWD) 기준 상대 경로입니다.
            CWD가 변경되면 경로가 달라질 수 있으므로 주의하세요.
        """
        self.logger = get_logger("gpt2_ivr.remap")
        self.distilled_tokenizer_path = Path(distilled_tokenizer_dir)
        self.remapped_tokenizer_path = Path(remapped_tokenizer_dir)
        self.remap_rules_path = Path(remap_rules_path)
        self.replacement_candidates_path = Path(replacement_candidates_path)

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """커맨드 실행 로직"""
        self.logger.info("Executing Remap Command...")

        # 1. Load distilled tokenizer
        if not self.distilled_tokenizer_path.exists():
            self.logger.error(
                "Distilled tokenizer not found at %s. "
                "Please run 'uv run ivr distill-tokenizer' first.",
                self.distilled_tokenizer_path,
            )
            return {"status": "failed", "message": "Distilled tokenizer missing"}

        self.logger.info(
            "Loading distilled tokenizer from %s", self.distilled_tokenizer_path
        )
        tokenizer = Tokenizer.from_file(
            str(self.distilled_tokenizer_path / "tokenizer.json")
        )

        # 2. Load replacement candidates (optional, for logging/info)
        if self.replacement_candidates_path.exists():
            candidates_df = pd.read_csv(self.replacement_candidates_path)
            self.logger.info(
                "Loaded %d replacement candidates from %s",
                len(candidates_df),
                self.replacement_candidates_path,
            )
            # self.logger.debug("Candidates: \n%s", candidates_df.head())
        else:
            self.logger.warning(
                "Replacement candidates file not found at %s. "
                "Skipping candidate-based logging.",
                self.replacement_candidates_path,
            )

        # 3. Load remap rules
        if not self.remap_rules_path.exists():
            self.logger.error(
                "Remap rules file not found at %s. Cannot perform remapping.",
                self.remap_rules_path,
            )
            return {"status": "failed", "message": "Remap rules missing"}

        with open(self.remap_rules_path, "r", encoding="utf-8") as f:
            remap_rules = yaml.safe_load(f)
            if not remap_rules:
                self.logger.warning(
                    "No remap rules found in %s.", self.remap_rules_path
                )
                remap_rules = {}
            self.logger.info(
                "Loaded %d remap rules from %s",
                len(remap_rules),
                self.remap_rules_path,
            )

        # 4. Apply remapping
        # This is a simplified example. Actual IVR remapping involves
        # more complex logic to manage token IDs and vocabulary.
        # For this implementation, we'll simulate adding new tokens and
        # potentially updating existing ones.
        current_vocab_size = tokenizer.get_vocab_size()
        self.logger.info("Current tokenizer vocab size: %d", current_vocab_size)

        new_tokens_to_add = []
        for old_token, new_token in remap_rules.items():
            old_id = tokenizer.token_to_id(old_token)
            new_id = tokenizer.token_to_id(new_token)

            if old_id is None and new_id is None:
                # Both are new, add new_token
                new_tokens_to_add.append(new_token)
                self.logger.info(
                    "Rule: '%s' -> '%s'. Both are new. Will add '%s' later.",
                    old_token,
                    new_token,
                    new_token,
                )
            elif old_id is not None and new_id is None:
                # Old token exists, new token is new.
                # In a real IVR, we'd assign old_id to new_token.
                # For this simplified example, we just add new_token.
                new_tokens_to_add.append(new_token)
                self.logger.info(
                    "Rule: '%s' (id:%d) -> '%s'. '%s' is new. Will add '%s' later.",
                    old_token,
                    old_id,
                    new_token,
                    new_token,
                    new_token,
                )
            elif old_id is None and new_id is not None:
                self.logger.warning(
                    "Rule: '%s' -> '%s' (id:%d). '%s' is new but target '%s' already exists. Skipping.",
                    old_token,
                    new_token,
                    new_id,
                    old_token,
                    new_token,
                )
            else:  # old_id is not None and new_id is not None
                self.logger.info(
                    "Rule: '%s' (id:%d) -> '%s' (id:%d). Both exist. No change for now.",
                    old_token,
                    old_id,
                    new_token,
                    new_id,
                )

        if new_tokens_to_add:
            self.logger.info(
                "Adding %d new tokens to the tokenizer.", len(new_tokens_to_add)
            )
            tokenizer.add_tokens(list(set(new_tokens_to_add)))  # Add unique new tokens
            self.logger.info(
                "New tokenizer vocab size after adding tokens: %d",
                tokenizer.get_vocab_size(),
            )
        else:
            self.logger.info("No new tokens to add based on remap rules.")

        # 5. Save the remapped tokenizer
        self.remapped_tokenizer_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(self.remapped_tokenizer_path / "tokenizer.json"))

        self.logger.info(
            "Remapped tokenizer saved to %s",
            self.remapped_tokenizer_path / "tokenizer.json",
        )
        self.logger.info("Remap Command finished.")
        return {
            "status": "success",
            "remapped_tokenizer_path": str(self.remapped_tokenizer_path),
        }

    def get_name(self) -> str:
        """커맨드 이름 반환"""
        return "remap"
