from __future__ import annotations

import argparse
import logging
from collections.abc import Callable

from pyfiglet import Figlet


CommandHandler = Callable[[logging.Logger], None]


def build_banner(text: str, font: str = "standard") -> str:
    """배너 문자열을 생성한다."""
    figlet = Figlet(font=font)
    return figlet.renderText(text)


def run_analyze(logger: logging.Logger) -> None:
    """분석 단계를 실행한다."""
    logger.info("[analyze] GPT-2 BPE 기준 토큰 시퀀스 분석을 시작합니다.")


def run_distill_tokenizer(logger: logging.Logger) -> None:
    """토크나이저 distillation 단계를 실행한다."""
    logger.info("[distill-tokenizer] BPE -> Unigram distillation 단계를 시작합니다.")


def run_select(logger: logging.Logger) -> None:
    """IVR 대상 선정 단계를 실행한다."""
    logger.info("[select] IVR 대상 토큰 선정을 시작합니다.")


def run_remap(logger: logging.Logger) -> None:
    """토큰 재할당 단계를 실행한다."""
    logger.info("[remap] 토큰 재할당 및 remap tokenizer 생성을 시작합니다.")


def run_align(logger: logging.Logger) -> None:
    """임베딩 정렬 단계를 실행한다."""
    logger.info("[align] embedding 재정렬 단계를 시작합니다.")


def run_train(logger: logging.Logger) -> None:
    """학습 단계를 실행한다."""
    logger.info("[train] 미세조정 단계를 시작합니다.")


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


def dispatch(command: str, logger: logging.Logger) -> None:
    """서브커맨드를 실행한다."""
    handlers: dict[str, CommandHandler] = {
        "analyze": run_analyze,
        "distill-tokenizer": run_distill_tokenizer,
        "select": run_select,
        "remap": run_remap,
        "align": run_align,
        "train": run_train,
    }
    handlers[command](logger)


def main() -> None:
    """CLI 엔트리 포인트."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("ivr")
    parser = build_parser()
    args = parser.parse_args()

    banner = build_banner("GPT2-IVR")
    print(banner)

    print_intro(logger)
    dispatch(args.command, logger)


if __name__ == "__main__":
    main()
