"""파이프라인 커맨드 추상 인터페이스.

파이프라인의 각 단계를 실행하는 커맨드 클래스들의 공통 인터페이스를 정의한다.
모든 커맨드는 Command 추상 클래스를 상속받아 execute()와 get_name()을 구현해야 한다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Command(ABC):
    """파이프라인 단계 실행 커맨드 인터페이스.

    모든 파이프라인 커맨드가 상속받아야 하는 추상 기반 클래스이다.
    각 커맨드는 execute()와 get_name() 메서드를 구현해야 한다.
    """

    @abstractmethod
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """커맨드 실행 로직.

        Args:
            **kwargs: 실행에 필요한 추가 인자

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
