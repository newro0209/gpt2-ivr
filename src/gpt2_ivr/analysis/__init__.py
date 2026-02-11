"""토큰 빈도 분석 및 IVR 후보 선정 모듈.

코퍼스를 토큰화하여 빈도를 분석하고, 저빈도 희생 토큰과 고빈도 바이그램 병합 후보를
매칭하여 IVR 교체 후보를 선정한다.
"""

from __future__ import annotations

from .candidate_selection import SelectionContext, select_replacement_candidates
from .token_frequency import analyze_token_frequency

__all__ = [
    "SelectionContext",
    "analyze_token_frequency",
    "select_replacement_candidates",
]
