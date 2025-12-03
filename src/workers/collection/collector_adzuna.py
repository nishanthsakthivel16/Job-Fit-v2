"""
Adzuna API Collector

Fetches job postings from Adzuna API with deduplication and date filtering.
Uses multithreading to speed up page fetching.

Compatible with:
- main_collection.py (AdzunaCollector(app_id=..., app_key=...))
- CollectorBase (fetch_jobs, _structure_job_data, get_statistics)

Enhanced with:
- Auto country injection per job
- Auto category inference from title/description when missing
"""

import os
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

from .collector_base import CollectorBase


class AdzunaCollector(CollectorBase):
    """
    Collector implementation for the Adzuna jobs API.

    main_collection.py expects:
        collector = AdzunaCollector(app_id=..., app_key=...)
        result = collector.fetch_jobs(...)
        stats  = collector.get_statistics()
    """

    BASE_URL = "https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"

    # Simple mapping from Adzuna country codes to human-friendly names
    COUNTRY_LABELS = {
        "gb": "UK",
        "us": "USA",
        "ca": "Canada",
        "au": "Australia",
        "in": "India",
        "sg": "Singapore",
        "za": "South Africa",
        "fr": "France",
        "de": "Germany",
        "nz": "New Zealand",
        "ie": "Ireland",
    }

    def __init__(
        self,
        app_id: str,
        app_key: str,
        data_dir: Optional[Path] = None,
        max_pages: int = 200,
        max_workers: int = 3,
    ):
        """
        Args:
            app_id: Adzuna APP ID
            app_key: Adzuna APP KEY
            data_dir: Optional root data directory (CollectorBase will default)
            max_pages: max pages per query (safety cap)
            max_workers: number of threads for parallel page fetching
        """
        super().__init__(data_dir=data_dir)

        # Prefer explicit values, fallback to env if needed
        self.app_id = app_id or os.getenv("ADZUNA_APP_ID")
        self.app_key = app_key or os.getenv("ADZUNA_APP_KEY")

        if not self.app_id or not self.app_key:
            raise ValueError(
                "AdzunaCollector requires app_id and app_key "
                "(or ADZUNA_APP_ID/ADZUNA_APP_KEY in env)."
            )

        self.max_pages = max_pages
        self.max_workers = max_workers

        # Will hold stats from the last fetch_jobs run
        self._last_stats: Dict[str, Any] = {
            "unique_jobs": 0,
            "total_collection_files": 0,
            "raw_data_directory": "",
        }

        # Used to inject country info into each job
        self._current_country_code: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_params(
        self,
        what: Optional[str],
        where: Optional[str],
        category: Optional[str],
        max_days_old: int,
        results_per_page: int,
    ) -> Dict[str, Any]:
        """Build query parameters for Adzuna API."""
        params: Dict[str, Any] = {
            "app_id": self.app_id,
            "app_key": self.app_key,
            "results_per_page": results_per_page,
            "content-type": "application/json",
        }
        if what:
            params["what"] = what
        if where:
            params["where"] = where
        if category:
            params["category"] = category

        if max_days_old is not None:
            params["max_days_old"] = max_days_old

        return params

    def _fetch_page(
        self,
        country: str,
        page: int,
        base_params: Dict[str, Any],
    ) -> List[Dict]:
        """Fetch a single page from Adzuna with timeout and logging."""
        url = self.BASE_URL.format(country=country, page=page)
        print(f"[Adzuna]   → Requesting page {page} for country={country}...")

        try:
            response = requests.get(url, params=base_params, timeout=20)
        except requests.RequestException as e:
            print(f"[Adzuna ❌] Request exception on page {page}: {e}")
            return []

        if response.status_code == 429:
            print(f"[Adzuna ⚠️] Rate limited (429) at page {page}")
            return []

        if response.status_code != 200:
            print(f"[Adzuna ❌] Request failed: {response.status_code} at page {page}")
            return []

        data = response.json()
        results = data.get("results", [])
        print(f"[Adzuna]   ← Page {page} returned {len(results)} jobs")

        return results

    @staticmethod
    def _infer_category_from_text(
        title: Optional[str],
        description: Optional[str],
        raw_category: Optional[str],
    ) -> str:
        """
        Infer a coarse job category from title/description when Adzuna category is missing.

        This is a simple keyword-based classifier, good enough for analytics + clustering.
        """
        if raw_category:
            # Use Adzuna's own category label when available
            return raw_category

        text = f"{title or ''} {description or ''}".lower()

        # IT / Engineering / Data
        if any(
            kw in text
            for kw in [
                "software engineer",
                "developer",
                "frontend",
                "backend",
                "full stack",
                "devops",
                "data scientist",
                "data engineer",
                "machine learning",
                "ml engineer",
                "python",
                "java",
                "golang",
                "c++",
                "react",
                "typescript",
                "cloud",
                "aws",
                "azure",
                "kubernetes",
            ]
        ):
            return "IT & Engineering"

        # Data / Analytics
        if any(
            kw in text
            for kw in [
                "data analyst",
                "business intelligence",
                "analytics",
                "bi analyst",
                "statistician",
            ]
        ):
            return "Data & Analytics"

        # Finance / Accounting
        if any(
            kw in text
            for kw in [
                "accountant",
                "finance",
                "financial analyst",
                "auditor",
                "tax",
                "bookkeeper",
                "controller",
                "fund",
                "banking",
            ]
        ):
            return "Finance & Accounting"

        # Marketing / Content
        if any(
            kw in text
            for kw in [
                "marketing",
                "digital marketing",
                "seo",
                "sem",
                "social media",
                "content writer",
                "copywriter",
                "brand",
                "growth marketer",
            ]
        ):
            return "Marketing & Content"

        # Sales / Business
        if any(
            kw in text
            for kw in [
                "sales",
                "account executive",
                "business development",
                "bdr",
                "sdr",
                "inside sales",
                "sales manager",
                "sales representative",
            ]
        ):
            return "Sales & Business Development"

        # HR / People
        if any(
            kw in text
            for kw in [
                "human resources",
                "hr manager",
                "recruiter",
                "talent acquisition",
                "people ops",
            ]
        ):
            return "HR & Recruitment"

        # Health / Medical
        if any(
            kw in text
            for kw in [
                "nurse",
                "doctor",
                "gp",
                "physician",
                "medical",
                "clinical",
                "hospital",
                "healthcare",
                "care assistant",
                "carer",
            ]
        ):
            return "Healthcare & Nursing"

        # Education / Teaching
        if any(
            kw in text
            for kw in [
                "teacher",
                "teaching assistant",
                "lecturer",
                "professor",
                "tutor",
                "education",
                "school",
                "academy",
            ]
        ):
            return "Education & Teaching"

        # Design / Creative
        if any(
            kw in text
            for kw in [
                "ux",
                "ui",
                "designer",
                "graphic design",
                "illustrator",
                "creative",
                "product design",
                "animation",
                "3d artist",
            ]
        ):
            return "Design & Creative"

        # Operations / Logistics
        if any(
            kw in text
            for kw in [
                "operations",
                "logistics",
                "supply chain",
                "warehouse",
                "procurement",
                "fleet",
            ]
        ):
            return "Operations & Logistics"

        # Management / PM
        if any(
            kw in text
            for kw in [
                "project manager",
                "programme manager",
                "product manager",
                "scrum master",
                "delivery manager",
            ]
        ):
            return "Project & Product Management"

        # Default
        return "Unknown"

    def _resolve_country(self, raw_job: Dict) -> str:
        """
        Decide the country label for a job.

        Priority:
        1) Use the country code passed to fetch_jobs (self._current_country_code)
        2) Try Adzuna location.area[0]
        3) Fallback to 'Unknown'
        """
        # 1) From current fetch context
        if self._current_country_code:
            code = self._current_country_code.lower()
            return self.COUNTRY_LABELS.get(code, code.upper())

        # 2) From job location area, if present
        area = raw_job.get("location", {}).get("area") or []
        if area:
            # Often looks like ["UK", "London", ...]
            return str(area[0])

        # 3) Fallback
        return "Unknown"

    # ------------------------------------------------------------------ #
    # Required interface for CollectorBase / main_collection
    # ------------------------------------------------------------------ #

    def fetch_jobs(
        self,
        country: str,
        what: Optional[str],
        where: Optional[str],
        category: Optional[str],
        max_days_old: int,
        max_jobs: int,
        results_per_page: int,
    ) -> Dict:
        """
        Fetch jobs from Adzuna using provided filters.

        This matches how main_collection.py calls it.

        Uses multithreading to fetch multiple pages in parallel.
        """
        print("\n================ ADZUNA COLLECTION ====================")
        print(f"  Country        : {country}")
        print(f"  What (keyword) : {what}")
        print(f"  Where (location): {where}")
        print(f"  Category       : {category}")
        print(f"  Max days old   : {max_days_old}")
        print(f"  Max jobs       : {max_jobs}")
        print(f"  Results / page : {results_per_page}")
        print(f"  Max pages      : {self.max_pages}")
        print(f"  Max workers    : {self.max_workers}")
        print("=======================================================\n")

        # Remember which country this run is for
        self._current_country_code = country

        start_time = datetime.now()   # for duration
        collected_jobs: List[Dict] = []
        seen_ids: Set[str] = set()

        total_api_results = 0
        api_calls = 0
        collection_time = datetime.now()

        base_params = self._build_params(
            what=what,
            where=where,
            category=category,
            max_days_old=max_days_old,
            results_per_page=results_per_page,
        )

        page = 1
        batch_size = self.max_workers  # number of pages fetched in parallel

        while len(collected_jobs) < max_jobs and page <= self.max_pages:
            pages_batch = list(range(page, min(page + batch_size, self.max_pages + 1)))
            if not pages_batch:
                break

            print(f"[Adzuna] Fetching pages batch: {pages_batch}")

            page_results: List[List[Dict]] = []

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._fetch_page, country, p, base_params): p
                    for p in pages_batch
                }
                api_calls += len(futures)  # how many HTTP calls we attempted in this batch

                for future in as_completed(futures):
                    p = futures[future]
                    results = future.result() or []
                    page_results.append(results)
                    total_api_results += len(results)

            # If all pages in this batch returned 0, we assume no more data
            if all(len(r) == 0 for r in page_results):
                print("[Adzuna] All pages in batch returned 0 results. Stopping.")
                break

            # Process results
            for results in page_results:
                for raw_job in results:
                    job_id = str(raw_job.get("id") or raw_job.get("adref") or "")
                    if not job_id:
                        continue

                    # Deduplicate within this run
                    if job_id in seen_ids:
                        continue

                    # Date filter
                    created = raw_job.get("created")
                    if created and max_days_old is not None:
                        try:
                            created_dt = datetime.fromisoformat(
                                created.replace("Z", "+00:00")
                            ).replace(tzinfo=None)
                            if created_dt < datetime.now() - timedelta(days=max_days_old):
                                continue
                        except Exception:
                            # If parsing fails, keep the job
                            pass

                    structured = self._structure_job_data(raw_job)
                    collected_jobs.append(structured)
                    seen_ids.add(job_id)

                    if len(collected_jobs) >= max_jobs:
                        break
                if len(collected_jobs) >= max_jobs:
                    break

            page += batch_size

        end_time = datetime.now()
        duration_seconds = (end_time - start_time).total_seconds()

        print(
            f"\n[Adzuna] Finished collection loop: {len(collected_jobs)} new jobs "
            f"(from {total_api_results} API results)."
        )

        # Use CollectorBase helper to save
        filepath = self.save_raw_list(collected_jobs, "adzuna")

        # Update last_stats so get_statistics() can report correctly
        self._last_stats = {
            "unique_jobs": len(collected_jobs),
            "total_collection_files": 1,
            "raw_data_directory": str(Path(filepath).parent),
        }

        print("\n================ ADZUNA COLLECTION SUMMARY ============")
        print(f"  Jobs collected      : {len(collected_jobs)}")
        print(f"  Approx. API results : {total_api_results}")
        print(f"  API calls attempted : {api_calls}")
        print(f"  Duration (seconds)  : {duration_seconds:.2f}")
        print(f"  Output file         : {filepath}")
        print("=======================================================\n")

        # status expected by main_collection:
        # - "success" when we collected jobs
        # - "no_new_jobs" when 0 collected
        status = "success" if len(collected_jobs) > 0 else "no_new_jobs"

        # Clear current country context
        self._current_country_code = None

        return {
            "timestamp": collection_time.isoformat(),
            "jobs_collected": len(collected_jobs),
            "jobs_deduplicated": len(collected_jobs),
            "filepath": str(filepath),
            "source": "adzuna",
            "api_calls": api_calls,
            "duration_seconds": duration_seconds,
            "status": status,
        }

    def _structure_job_data(self, raw_job: Dict) -> Dict:
        """
        Convert raw Adzuna job JSON into a standardized format.

        Enhanced with:
        - country: inferred from fetch context or location.area
        - category: inferred from title/description if Adzuna category is missing
        """
        raw_category_label = (raw_job.get("category") or {}).get("label")
        title = raw_job.get("title")
        description = raw_job.get("description")

        inferred_category = self._infer_category_from_text(
            title=title,
            description=description,
            raw_category=raw_category_label,
        )
        country_label = self._resolve_country(raw_job)

        return {
            "id": raw_job.get("id") or raw_job.get("adref"),
            "title": title,
            "company": raw_job.get("company", {}).get("display_name"),
            "location": raw_job.get("location", {}).get("display_name"),
            "country": country_label,
            "description": description,
            "category": inferred_category,
            "created": raw_job.get("created"),
            "redirect_url": raw_job.get("redirect_url"),
            "salary_min": raw_job.get("salary_min"),
            "salary_max": raw_job.get("salary_max"),
            "contract_type": raw_job.get("contract_type"),
            "contract_time": raw_job.get("contract_time"),
            "source": "adzuna",
            "collected_at": datetime.now().isoformat(),
        }

    def get_statistics(self) -> Dict:
        """
        Return statistics for the last collection run.

        main_collection.py expects:
            stats["unique_jobs"]
            stats["total_collection_files"]
            stats["raw_data_directory"]
        """
        return self._last_stats or {
            "unique_jobs": 0,
            "total_collection_files": 0,
            "raw_data_directory": "",
        }
