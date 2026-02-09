"""중앙화된 로깅/진행바 설정"""

import logging
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from gpt2_ivr.constants import LOGS_DIR

_CONSOLE = Console(stderr=False)


def get_console() -> Console:
    """로깅과 진행바에서 공용으로 사용할 Rich 콘솔을 반환한다."""
    return _CONSOLE


def create_progress(*, transient: bool = False, disable: bool = False) -> Progress:
    """일반 작업용 Rich 진행바를 생성한다."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=get_console(),
        transient=transient,
        disable=disable,
    )


def create_byte_progress(*, transient: bool = False, disable: bool = False) -> Progress:
    """바이트 단위 처리 작업용 Rich 진행바를 생성한다."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        DownloadColumn(binary_units=False),
        TransferSpeedColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=get_console(),
        transient=transient,
        disable=disable,
    )


def setup_logging(
    level: int = logging.INFO,
    format_string: str = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    log_to_file: bool = True,
    log_dir: Path | None = None,
) -> None:
    """전역 로깅을 설정한다.

    애플리케이션 시작 시 한 번만 호출해야 한다.
    중복 핸들러 생성을 방지한다.

    Args:
        level: 로깅 레벨
        format_string: 로그 포맷 문자열
        log_to_file: 파일로 로그를 저장할지 여부
        log_dir: 로그 파일 저장 디렉토리 (None이면 LOGS_DIR 상수 사용)
    """
    root_logger = logging.getLogger()

    # 이미 핸들러가 있으면 설정 완료
    if root_logger.handlers:
        return

    root_logger.setLevel(level)

    # Rich 콘솔 핸들러 설정
    console_handler = RichHandler(
        console=get_console(),
        show_path=False,
        rich_tracebacks=True,
        markup=False,
        log_time_format="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(console_handler)

    # 파일 핸들러 설정
    if log_to_file:
        if log_dir is None:
            log_dir = LOGS_DIR
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"ivr_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(format_string))
        root_logger.addHandler(file_handler)

        # 로그 파일 경로 출력
        root_logger.info("📝 로그 파일: %s", log_file)


def get_logger(name: str) -> logging.Logger:
    """모듈별 로거를 반환한다.

    Args:
        name: 로거 이름 (보통 __name__ 사용)

    Returns:
        설정된 로거 인스턴스
    """
    return logging.getLogger(name)
