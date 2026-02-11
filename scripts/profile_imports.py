"""모듈 import 병목을 측정하고 요약 리포트를 생성한다."""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


IMPORTTIME_LINE_PATTERN = re.compile(r"^import time:\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(.+)$")


@dataclass(slots=True)
class ImportRecord:
    """단일 importtime 레코드를 표현한다.

    Attributes:
        module: 모듈 이름
        self_us: 모듈 자체 import 시간(마이크로초)
        cumulative_us: 누적 import 시간(마이크로초)
    """

    module: str
    self_us: int
    cumulative_us: int


@dataclass(slots=True)
class ModuleSummary:
    """모듈별 반복 측정 집계 정보를 표현한다.

    Attributes:
        module: 모듈 이름
        samples: 샘플 수
        self_avg_us: 자체 시간 평균(마이크로초)
        self_median_us: 자체 시간 중앙값(마이크로초)
        self_max_us: 자체 시간 최대값(마이크로초)
        cumulative_avg_us: 누적 시간 평균(마이크로초)
        cumulative_median_us: 누적 시간 중앙값(마이크로초)
        cumulative_max_us: 누적 시간 최대값(마이크로초)
    """

    module: str
    samples: int
    self_avg_us: float
    self_median_us: float
    self_max_us: int
    cumulative_avg_us: float
    cumulative_median_us: float
    cumulative_max_us: int


def parse_args() -> argparse.Namespace:
    """커맨드라인 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="모듈 import 병목을 반복 측정하고 상위 지연 구간을 요약한다.")
    parser.add_argument("--target", default="gpt2_ivr.cli", help="측정할 import 대상 모듈")
    parser.add_argument("--iterations", type=int, default=5, help="반복 측정 횟수")
    parser.add_argument("--top", type=int, default=20, help="요약에 출력할 상위 모듈 수")
    parser.add_argument("--timeout", type=int, default=180, help="단일 측정 최대 대기 시간(초)")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/analysis/reports"), help="리포트 저장 디렉토리")
    return parser.parse_args()


def build_pythonpath() -> str:
    """`src`를 포함한 PYTHONPATH를 구성한다."""
    src_path = str(Path("src").resolve())
    existing = os.environ.get("PYTHONPATH")
    if not existing:
        return src_path
    return os.pathsep.join([src_path, existing])


def parse_importtime(stderr_text: str) -> list[ImportRecord]:
    """importtime stderr를 파싱해 레코드 목록으로 변환한다.

    Args:
        stderr_text: `python -X importtime` 표준 에러 출력

    Returns:
        파싱된 ImportRecord 목록
    """
    records: list[ImportRecord] = []
    for line in stderr_text.splitlines():
        match = IMPORTTIME_LINE_PATTERN.match(line)
        if not match:
            continue
        self_us, cumulative_us, module_raw = match.groups()
        records.append(
            ImportRecord(
                module=module_raw.strip(),
                self_us=int(self_us),
                cumulative_us=int(cumulative_us),
            )
        )
    return records


def measure_once(target: str, timeout: int) -> tuple[list[ImportRecord], str]:
    """대상 모듈 import를 1회 측정한다."""
    cmd = [sys.executable, "-X", "importtime", "-c", f"import {target}"]
    env = dict(os.environ)
    env["PYTHONPATH"] = build_pythonpath()
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "importtime 실행 실패\n"
            f"- 반환 코드: {completed.returncode}\n"
            f"- stderr:\n{completed.stderr.strip()}"
        )
    return parse_importtime(completed.stderr), completed.stderr


def summarize(all_records: list[list[ImportRecord]]) -> list[ModuleSummary]:
    """반복 측정 결과를 모듈별로 집계한다."""
    by_module: dict[str, dict[str, list[int]]] = {}
    for records in all_records:
        for rec in records:
            bucket = by_module.setdefault(rec.module, {"self": [], "cumulative": []})
            bucket["self"].append(rec.self_us)
            bucket["cumulative"].append(rec.cumulative_us)

    summaries: list[ModuleSummary] = []
    for module, values in by_module.items():
        self_values = values["self"]
        cumulative_values = values["cumulative"]
        summaries.append(
            ModuleSummary(
                module=module,
                samples=len(self_values),
                self_avg_us=statistics.mean(self_values),
                self_median_us=statistics.median(self_values),
                self_max_us=max(self_values),
                cumulative_avg_us=statistics.mean(cumulative_values),
                cumulative_median_us=statistics.median(cumulative_values),
                cumulative_max_us=max(cumulative_values),
            )
        )
    summaries.sort(key=lambda item: item.cumulative_avg_us, reverse=True)
    return summaries


def write_report(
    summaries: list[ModuleSummary],
    target: str,
    iterations: int,
    top_n: int,
    summary_path: Path,
    raw_path: Path,
) -> None:
    """요약 마크다운 리포트를 작성한다."""
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    top_rows = summaries[:top_n]
    lines = [
        "# Import 병목 요약",
        "",
        f"- 생성 시각: {generated_at}",
        f"- 대상 모듈: `{target}`",
        f"- 반복 횟수: {iterations}",
        f"- 원시 로그: `{raw_path}`",
        "",
        "| 순위 | 모듈 | 샘플 | 자기 평균(ms) | 누적 평균(ms) | 누적 중앙값(ms) | 누적 최대(ms) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(top_rows, start=1):
        lines.append(
            "| "
            f"{idx} | `{row.module}` | {row.samples} | "
            f"{row.self_avg_us / 1000:.3f} | {row.cumulative_avg_us / 1000:.3f} | "
            f"{row.cumulative_median_us / 1000:.3f} | {row.cumulative_max_us / 1000:.3f} |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_raw_log(chunks: list[str], output_path: Path) -> None:
    """반복 측정의 원시 stderr 로그를 저장한다."""
    sections: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        sections.append(f"===== iteration {index} =====")
        sections.append(chunk.rstrip())
        sections.append("")
    output_path.write_text("\n".join(sections), encoding="utf-8")


def print_top(summaries: list[ModuleSummary], top_n: int) -> None:
    """상위 병목 모듈을 콘솔에 출력한다."""
    print("상위 import 병목 (누적 평균 기준)")
    print("순위 | 모듈 | 누적 평균(ms) | 누적 중앙값(ms) | 누적 최대(ms)")
    for idx, row in enumerate(summaries[:top_n], start=1):
        print(
            f"{idx:>2} | {row.module} | {row.cumulative_avg_us / 1000:.3f} | "
            f"{row.cumulative_median_us / 1000:.3f} | {row.cumulative_max_us / 1000:.3f}"
        )


def main() -> int:
    """스크립트 엔트리 포인트."""
    args = parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations는 1 이상이어야 한다.")
    if args.top < 1:
        raise ValueError("--top은 1 이상이어야 한다.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_label = args.target.replace(".", "_")
    raw_path = args.output_dir / f"importtime_{target_label}_{timestamp}.log"
    summary_path = args.output_dir / f"importtime_{target_label}_{timestamp}_summary.md"

    all_records: list[list[ImportRecord]] = []
    raw_chunks: list[str] = []
    for _ in range(args.iterations):
        records, stderr = measure_once(target=args.target, timeout=args.timeout)
        all_records.append(records)
        raw_chunks.append(stderr)

    summaries = summarize(all_records)
    write_raw_log(raw_chunks, raw_path)
    write_report(
        summaries=summaries,
        target=args.target,
        iterations=args.iterations,
        top_n=args.top,
        summary_path=summary_path,
        raw_path=raw_path,
    )
    print_top(summaries, top_n=args.top)
    print(f"\n요약 리포트: {summary_path}")
    print(f"원시 로그: {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
