from __future__ import annotations

import argparse
import logging

from pyfiglet import Figlet

from gpt2_ivr.commands import (
    AnalyzeCommand,
    Command,
    DistillCommand,
    SelectCommand,
    RemapCommand,
    AlignCommand,
    TrainCommand,
)
from gpt2_ivr.utils.logging_config import get_logger, setup_logging


def build_banner(text: str, font: str = "standard") -> str:
    """배너 문자열을 생성한다."""
    figlet = Figlet(font=font)
    return figlet.renderText(text)


def build_parser() -> argparse.ArgumentParser:
    """IVR CLI 파서를 생성한다."""
    parser = argparse.ArgumentParser(
        prog="ivr",
        description="Tokenizer Model Migration + IVR 파이프라인 CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("analyze", help="BPE 토큰 시퀀스 분석")
    subparsers.add_parser("distill-tokenizer", help="BPE -> Unigram distillation")
    subparsers.add_parser("select", help="IVR 대상 토큰 선정")
    subparsers.add_parser("remap", help="토큰 재할당 규칙 적용")
    subparsers.add_parser("align", help="embedding 재정렬")
    subparsers.add_parser("train", help="미세조정")
    return parser


def print_intro(logger: logging.Logger) -> None:
    """실행 개요를 출력한다."""
    logger.info("Tokenizer Model Migration + IVR 파이프라인")
    logger.info("BPE -> Unigram 토크나이저 교체 후 IVR를 수행합니다.")


def create_command(command_name: str) -> Command:
    """커맨드 이름에 해당하는 Command 객체를 생성한다."""
    commands: dict[str, type[Command]] = {
        "analyze": AnalyzeCommand,
        "distill-tokenizer": DistillCommand,
        "select": SelectCommand,
        "remap": RemapCommand,
        "align": AlignCommand,
        "train": TrainCommand,
    }

    if command_name in commands:
        return commands[command_name]()

    raise NotImplementedError(f"'{command_name}'는 유효하지 않은 커맨드입니다.")


def dispatch(command_name: str, logger: logging.Logger) -> None:
    """서브커맨드를 실행한다."""
    logger.info("[%s] 단계를 시작합니다.", command_name)

    try:
        command = create_command(command_name)
        result = command.execute()
        logger.info("[%s] 단계 완료. 결과: %s", command_name, result)
    except NotImplementedError as e:
        logger.warning("%s", e)
    except Exception as e:
        logger.error("[%s] 실행 중 오류 발생: %s", command_name, e)
        raise


def main() -> None:
    """CLI 엔트리 포인트."""
    setup_logging()
    logger = get_logger("gpt2_ivr.cli")
    
    parser = build_parser()
    args = parser.parse_args()

    banner = build_banner("GPT2-IVR")
    print(banner)

    print_intro(logger)
    dispatch(args.command, logger)


if __name__ == "__main__":
    main()
