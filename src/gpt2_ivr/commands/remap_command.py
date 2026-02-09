"""토크나이저 재할당 커맨드.

재할당 규칙(YAML)을 증류된 Unigram 토크나이저에 적용하여
희생 토큰을 신규 토큰으로 교체한 재할당 토크나이저를 생성한다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from rich.console import Console
from rich.panel import Panel
from tokenizers import Tokenizer

from gpt2_ivr.commands.base import Command

console = Console()


class RemapCommand(Command):
    """토큰 재할당 규칙 적용 커맨드.

    YAML 재할당 규칙에 따라 증류 토크나이저에 신규 토큰을 추가하고
    재할당된 토크나이저를 저장한다.

    Attributes:
        logger: 로거 인스턴스
        distilled_tokenizer_path: 증류된 토크나이저 디렉토리
        remapped_tokenizer_path: 재할당 토크나이저 저장 디렉토리
        remap_rules_path: 재할당 규칙 YAML 파일 경로
        replacement_candidates_path: 교체 후보 CSV 경로
    """

    def __init__(
        self,
        distilled_tokenizer_dir: Path,
        remapped_tokenizer_dir: Path,
        remap_rules_path: Path,
        replacement_candidates_path: Path,
    ) -> None:
        self.logger = logging.getLogger("gpt2_ivr.remap")
        self.distilled_tokenizer_path = distilled_tokenizer_dir
        self.remapped_tokenizer_path = remapped_tokenizer_dir
        self.remap_rules_path = remap_rules_path
        self.replacement_candidates_path = replacement_candidates_path

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """토큰 재할당을 실행한다.

        증류 토크나이저에 재할당 규칙을 적용하여 신규 토큰을 추가하고
        재할당 토크나이저를 저장한다.

        Args:
            **kwargs: 사용되지 않음

        Returns:
            실행 결과 딕셔너리 (status, remapped_tokenizer_path)

        Raises:
            FileNotFoundError: 증류 토크나이저 또는 재할당 규칙 파일이 없는 경우
            ValueError: 재할당 규칙 형식이 올바르지 않은 경우
        """
        # 1. 증류 토크나이저 로드
        if not self.distilled_tokenizer_path.exists():
            raise FileNotFoundError(
                "증류 토크나이저를 찾을 수 없습니다. "
                f"먼저 `uv run ivr distill-tokenizer`를 실행하세요: "
                f"{self.distilled_tokenizer_path}"
            )

        self.logger.info("증류 토크나이저 로드: %s", self.distilled_tokenizer_path)
        tokenizer = Tokenizer.from_file(
            str(self.distilled_tokenizer_path / "tokenizer.json")
        )

        # 2. 교체 후보 로드 (선택, 로그 정보용)
        if self.replacement_candidates_path.exists():
            candidates_df = pd.read_csv(self.replacement_candidates_path)
            self.logger.info(
                "교체 후보 %d개 로드: %s",
                len(candidates_df),
                self.replacement_candidates_path,
            )
            # self.logger.debug("교체 후보 샘플:\n%s", candidates_df.head())
        else:
            self.logger.warning(
                "교체 후보 CSV가 없어 상세 로그를 생략합니다: %s",
                self.replacement_candidates_path,
            )

        # 3. 재할당 규칙 로드
        if not self.remap_rules_path.exists():
            raise FileNotFoundError(
                "재할당 규칙 파일을 찾을 수 없습니다: " f"{self.remap_rules_path}"
            )

        with self.remap_rules_path.open("r", encoding="utf-8") as handle:
            loaded_rules = yaml.safe_load(handle)

        if loaded_rules is None:
            self.logger.warning(
                "재할당 규칙이 비어 있습니다: %s", self.remap_rules_path
            )
            remap_rules: dict[str, str] = {}
        elif isinstance(loaded_rules, dict):
            remap_rules = loaded_rules
        else:
            raise ValueError(
                "재할당 규칙 형식이 올바르지 않습니다. "
                "YAML 매핑(dict) 형식이어야 합니다."
            )

        self.logger.info(
            "재할당 규칙 %d개 로드: %s",
            len(remap_rules),
            self.remap_rules_path,
        )

        # 4. 재할당 규칙 적용
        # 현재 구현은 단순화된 형태이며, 실제 IVR에서는
        # 토큰 ID/어휘 재배치를 더 정교하게 다룰 필요가 있다.
        # 여기서는 신규 토큰 추가 중심으로 동작한다.
        current_vocab_size = tokenizer.get_vocab_size()
        self.logger.info("현재 토크나이저 vocab 크기: %d", current_vocab_size)

        new_tokens_to_add = []
        for old_token, new_token in remap_rules.items():
            old_id = tokenizer.token_to_id(old_token)
            new_id = tokenizer.token_to_id(new_token)

            if old_id is None and new_id is None:
                # 양쪽 모두 신규 토큰인 경우
                new_tokens_to_add.append(new_token)
                self.logger.info(
                    "규칙: '%s' -> '%s' (신규 토큰 추가 예정)",
                    old_token,
                    new_token,
                )
            elif old_id is not None and new_id is None:
                # 기존 토큰은 있고 신규 토큰은 없는 경우
                new_tokens_to_add.append(new_token)
                self.logger.info(
                    "규칙: '%s'(id:%d) -> '%s' (신규 토큰 추가 예정)",
                    old_token,
                    old_id,
                    new_token,
                )
            elif old_id is None and new_id is not None:
                self.logger.warning(
                    "규칙 무시: '%s' -> '%s'(id:%d), 대상 토큰이 이미 존재합니다.",
                    old_token,
                    new_token,
                    new_id,
                )
            else:  # 양쪽 모두 기존 토큰인 경우
                self.logger.info(
                    "규칙: '%s'(id:%d) -> '%s'(id:%d), 기존 토큰 유지",
                    old_token,
                    old_id,
                    new_token,
                    new_id,
                )

        if new_tokens_to_add:
            self.logger.info("신규 토큰 %d개를 추가합니다.", len(new_tokens_to_add))
            # 중복 제거 후 신규 토큰 추가
            tokenizer.add_tokens(list(set(new_tokens_to_add)))
            self.logger.info(
                "토큰 추가 후 vocab 크기: %d",
                tokenizer.get_vocab_size(),
            )
        else:
            self.logger.info("추가할 신규 토큰이 없습니다.")

        # 5. 재할당 토크나이저 저장
        self.remapped_tokenizer_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(self.remapped_tokenizer_path / "tokenizer.json"))

        # Rich 패널로 결과 출력
        result_text = f"""[bold cyan]재할당 토크나이저 저장 완료[/bold cyan]

[yellow]경로:[/yellow] {self.remapped_tokenizer_path / "tokenizer.json"}
[yellow]이전 vocab 크기:[/yellow] {current_vocab_size:,}
[yellow]현재 vocab 크기:[/yellow] {tokenizer.get_vocab_size():,}
[yellow]추가된 토큰:[/yellow] {len(new_tokens_to_add):,}개"""

        console.print()
        console.print(Panel(result_text, title="토큰 재할당 완료", border_style="green"))
        console.print()

        return {
            "status": "success",
            "remapped_tokenizer_path": str(self.remapped_tokenizer_path),
        }

    def get_name(self) -> str:
        """커맨드 이름을 반환한다.

        Returns:
            커맨드 이름 "remap"
        """
        return "remap"
