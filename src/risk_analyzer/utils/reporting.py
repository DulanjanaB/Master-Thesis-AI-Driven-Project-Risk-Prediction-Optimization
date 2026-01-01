from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class RunArtifactPaths:
    report_path: Path


def utc_timestamp_slug() -> str:
    dt = datetime.now(timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%SZ")


def write_text_report(*, report_dir: Path, stem: str, content: str) -> RunArtifactPaths:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{stem}.md"
    path.write_text(content, encoding="utf-8")
    return RunArtifactPaths(report_path=path)
