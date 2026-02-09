"""파이프라인 커맨드 추상 인터페이스"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Command(ABC):
    """파이프라인 단계 실행 커맨드 인터페이스"""

    @abstractmethod
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """커맨드 실행 로직

        Returns:
            실행 결과 딕셔너리 (다음 단계 입력용)
        """
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> str:
        """커맨드 이름 반환"""
        raise NotImplementedError
