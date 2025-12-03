"""
RemoteOK API Collector

Fetches remote job postings from the public RemoteOK API.

Compatible with:
- scripts/run_collection_remoteok.py
- CollectorBase (save_raw_list, get_statistics)
"""

import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

from .collector_base import CollectorBase


class RemoteOKCollector(CollectorBase):
    """
    Collector implementation for the RemoteOK jobs API.

    scripts/run_collection_remoteok.py expects:
        collector = RemoteOKCollector()
        result = collector.fetch_jobs(
            max_jobs=100,
            filter_tags=['python', 'react', 'javascript', 'engineer']  # or None
        )
        stats = collector.get_statistics()
    """

    BASE_URL = "https://remoteok.com/api"

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Args:
            data_dir: Optional root data directory (CollectorBase will default)
        """
        super().__init__(data_dir=data_dir)

        # Will hold stats from the last fetch_jobs run
        self._last_stats: Dict[str, Any] = {
            "unique_jobs": 0,
            "total_collection_files": 0,
            "raw_data_directory": "",
        }

    # ------------------------------------------------------------------ #
    # Core API call
    # ------------------------------------------------------------------ #

    def _fetch_all_jobs(self) -> List[Dict]:
        """
        Fetch all jobs from RemoteOK API.
        RemoteOK returns a JSON list; first element is usually metadata.
        """
        print(f"[RemoteOK]   → Requesting jobs from {self.BASE_URL} ...")

        headers = {
            # Optional header per RemoteOK docs (good practice)
            "Accept": "application/json",
            "User-Agent": "JobFit-AI-Project/1.0 (contact: your-team@example.com)",
        }

        response = requests.get(self.BASE_URL, headers=headers, timeout=20)

        if response.status_code != 200:
            print(f"[RemoteOK ❌] Request failed with status {response.status_code}")
            return []

        data = response.json()
        if not isinstance(data, list):
            print("[RemoteOK ❌] Unexpected response format (not a list)")
            return []

        # First element is usually API metadata
        jobs = data[1:]
        print(f"[RemoteOK]   ← Received {len(jobs)} jobs from API")
        return jobs

    def _filter_and_structure(
        self,
        jobs: List[Dict],
        max_jobs: int,
        filter_tags: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Filter by tags and limit to max_jobs, and structure data for downstream use.
        """
        normalized_tags = {t.lower() for t in (filter_tags or [])}
        collected: List[Dict] = []
        seen_ids: Set[str] = set()

        for job in jobs:
            job_id = str(job.get("id") or job.get("slug") or "")
            if not job_id:
                continue

            if job_id in seen_ids:
                continue

            # If filter_tags provided, only keep jobs that have at least one matching tag
            if normalized_tags:
                job_tags = {str(t).lower() for t in (job.get("tags") or [])}
                if job_tags.isdisjoint(normalized_tags):
                    continue

            structured = {
                "id": job_id,
                "title": job.get("position") or job.get("title"),
                "company": job.get("company"),
                "location": job.get("location") or "Remote",
                "description": job.get("description"),
                "tags": job.get("tags"),
                "url": job.get("url"),
                "created_at": job.get("date") or job.get("created_at"),
                "source": "remoteok",
                "collected_at": datetime.now().isoformat(),
            }

            collected.append(structured)
            seen_ids.add(job_id)

            if len(collected) >= max_jobs:
                break

        print(f"[RemoteOK]   → Filtered down to {len(collected)} jobs")
        return collected

    # ------------------------------------------------------------------ #
    # Public interface expected by run_collection_remoteok.py
    # ------------------------------------------------------------------ #

    def fetch_jobs(
        self,
        max_jobs: int = 100,
        filter_tags: Optional[List[str]] = None,
    ) -> Dict:
        """
        Fetch jobs from RemoteOK, optionally filtered by tags, up to max_jobs.

        Called as:
            collector.fetch_jobs(max_jobs=100, filter_tags=[...])
        """
        print("\n================ REMOTEOK COLLECTION ==================")
        print(f"  Max jobs       : {max_jobs}")
        print(f"  Filter tags    : {filter_tags}")
        print("  API endpoint   : https://remoteok.com/api")
        print("=======================================================\n")

        start_time = datetime.now()

        # 1. Fetch all jobs from RemoteOK
        jobs = self._fetch_all_jobs()
        api_calls = 1  # Single API call

        # 2. Filter and structure
        collected_jobs = self._filter_and_structure(
            jobs=jobs,
            max_jobs=max_jobs,
            filter_tags=filter_tags,
        )

        # 3. Save to disk
        filepath = self.save_raw_list(collected_jobs, "remoteok")

        end_time = datetime.now()
        duration_seconds = (end_time - start_time).total_seconds()

        # 4. Update stats for get_statistics()
        self._last_stats = {
            "unique_jobs": len(collected_jobs),
            "total_collection_files": 1,
            "raw_data_directory": str(Path(filepath).parent),
        }

        status = "success" if len(collected_jobs) > 0 else "no_new_jobs"

        print("\n================ REMOTEOK COLLECTION SUMMARY ==========")
        print(f"  Jobs collected      : {len(collected_jobs)}")
        print(f"  API calls           : {api_calls}")
        print(f"  Duration (seconds)  : {duration_seconds:.2f}")
        print(f"  Output file         : {filepath}")
        print("=======================================================\n")

        # IMPORTANT: scripts/run_collection_remoteok.py expects result["parameters"]
        return {
            "timestamp": start_time.isoformat(),
            "jobs_collected": len(collected_jobs),
            "jobs_deduplicated": len(collected_jobs),
            "filepath": str(filepath),
            "source": "remoteok",
            "api_calls": api_calls,
            "duration_seconds": duration_seconds,
            "status": status,
            "parameters": {          # ✅ so result["parameters"].get("filter_tags") works
                "max_jobs": max_jobs,
                "filter_tags": filter_tags,
            },
        }

    def get_statistics(self) -> Dict:
        """
        Return statistics for the last RemoteOK collection run.

        scripts/run_collection_remoteok.py likely prints:
            stats["unique_jobs"]
            stats["total_collection_files"]
            stats["raw_data_directory"]
        """
        return self._last_stats or {
            "unique_jobs": 0,
            "total_collection_files": 0,
            "raw_data_directory": "",
        }
