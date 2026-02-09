"""중앙화된 로깅 설정"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(
    level: int = logging.INFO,
    format_string: str = "[%(levelname)s] %(message)s",
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
        log_dir: 로그 파일 저장 디렉토리 (None이면 artifacts/logs/ 사용)
    """
    root_logger = logging.getLogger()

    # 이미 핸들러가 있으면 설정 완료
    if root_logger.handlers:
        return

    formatter = logging.Formatter(format_string)

    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 파일 핸들러 설정
    if log_to_file:
        if log_dir is None:
            log_dir = Path("artifacts/logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"ivr_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # 로그 파일 경로 출력
        root_logger.info(f"📝 로그 파일: {log_file}")

    root_logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """모듈별 로거를 반환한다.

    Args:
        name: 로거 이름 (보통 __name__ 사용)

    Returns:
        설정된 로거 인스턴스
    """
    return logging.getLogger(name)
