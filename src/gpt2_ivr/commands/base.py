"""파이프라인 커맨드 추상 인터페이스.

파이프라인의 각 단계를 실행하는 커맨드 클래스들의 공통 인터페이스를 정의한다.
모든 커맨드는 Command 추상 클래스를 상속받아 execute()와 get_name()을 구현해야 한다.
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from typing import Any, Protocol


class SubparsersLike(Protocol):
    """argparse 서브파서 액션 호환 프로토콜."""

    def add_parser(self, name: str, **kwargs: Any) -> argparse.ArgumentParser:
        """서브커맨드 파서를 추가한다."""
        ...


class Command(ABC):
    """파이프라인 단계 실행 커맨드 인터페이스.

    모든 파이프라인 커맨드가 상속받아야 하는 추상 기반 클래스이다.
    각 커맨드는 configure_parser(), execute(), get_name()을 구현해야 한다.
    """

    @staticmethod
    @abstractmethod
    def configure_parser(subparsers: SubparsersLike) -> None:
        """서브커맨드 파서를 설정한다.

        Args:
            subparsers: 서브파서 액션 객체
        """
        raise NotImplementedError

    @abstractmethod
    def execute(self) -> dict[str, Any]:
        """커맨드 실행 로직.

        Returns:
            실행 결과 딕셔너리 (다음 단계 입력용)

        Raises:
            NotImplementedError: 서브클래스에서 구현되지 않은 경우
        """
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> str:
        """커맨드 이름을 반환한다.

        Returns:
            커맨드의 고유 식별 이름

        Raises:
            NotImplementedError: 서브클래스에서 구현되지 않은 경우
        """
        raise NotImplementedError
