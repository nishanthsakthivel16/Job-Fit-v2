"""
Base collector for all job sources.
Handles:
✓ Folder setup
✓ Saving raw job files
"""

import json
from datetime import datetime
from pathlib import Path


class CollectorBase:
    """Base class for data collectors."""

    def __init__(self, data_dir: Path = None):
        # Root data folder
        if data_dir is None:
            self.data_dir = Path(__file__).resolve().parents[3] / "data"
        else:
            self.data_dir = data_dir

        # Raw directory for all collectors
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Save a SINGLE job into a raw JSONL file
    # ---------------------------------------------------------
    def save_raw_job(self, job: dict):
        """Append a single structured job to a daily JSONL file."""

        date_folder = self.raw_dir / job.get("source", "unknown") / datetime.now().strftime("%Y-%m-%d")
        date_folder.mkdir(parents=True, exist_ok=True)

        filepath = date_folder / "jobs.jsonl"

        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(job, ensure_ascii=False) + "\n")

        return filepath

    # ---------------------------------------------------------
    # Save MANY jobs at once (as JSON list)
    # ---------------------------------------------------------
    def save_raw_list(self, jobs: list, source: str):
        """Save a full list of jobs into a json file."""

        date_folder = self.raw_dir / source / datetime.now().strftime("%Y-%m-%d")
        date_folder.mkdir(parents=True, exist_ok=True)

        filename = f"jobs_{datetime.now().strftime('%H%M%S')}.json"
        filepath = date_folder / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(jobs, f, ensure_ascii=False, indent=2)

        return filepath
