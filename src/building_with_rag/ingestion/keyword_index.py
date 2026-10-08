"""Create (or reuse) the Atlas Search keyword index on chunks.text.

Run: uv run python -m building_with_rag.ingestion.keyword_index
Idempotent; never replaces a differing index; no Voyage calls, no data writes.
"""

import sys
import time

from pymongo import MongoClient
from pymongo.errors import PyMongoError
from pymongo.operations import SearchIndexModel

from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.settings import get_settings

WAIT_SECONDS = 120


def differences(existing: dict, wanted: dict, path: str = "") -> list[str]:
    """List wanted keys missing or different in *existing* (Atlas may add extra defaults)."""
    diffs = []
    for key, value in wanted.items():
        here = f"{path}.{key}" if path else key
        if key not in existing:
            diffs.append(f"{here}: missing")
        elif isinstance(value, dict):
            if not isinstance(existing[key], dict):
                diffs.append(f"{here}: expected an object")
            else:
                diffs.extend(differences(existing[key], value, here))
        elif existing[key] != value:
            diffs.append(f"{here}: expected {value!r}, found {existing[key]!r}")
    return diffs


def _find(coll):
    return next(
        (i for i in coll.list_search_indexes() if i.get("name") == schema.KEYWORD_INDEX_NAME), None
    )


def main() -> int:
    settings = get_settings()
    if not settings.mongodb_uri:
        print("MONGODB_URI is empty. Set it in .env and re-run.")
        return 1
    client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=8000)
    db_name = settings.mongodb_test_db_name if settings.app_env == "testing" else settings.mongodb_db_name
    coll = client[db_name][schema.CHUNKS_COLLECTION]
    try:
        index = _find(coll)
        if index is None:
            coll.create_search_index(SearchIndexModel(
                name=schema.KEYWORD_INDEX_NAME, type="search",
                definition=schema.KEYWORD_INDEX_DEFINITION,
            ))
            print(f"{schema.KEYWORD_INDEX_NAME}: created")
        else:
            existing = index.get("latestDefinition") or index.get("definition") or {}
            diffs = differences(existing, schema.KEYWORD_INDEX_DEFINITION)
            if diffs:
                print(f"{schema.KEYWORD_INDEX_NAME} exists with a different definition:")
                for d in diffs:
                    print(f"  - {d}")
                print("Drop it manually in Atlas and re-run; it is never replaced automatically.")
                return 1
            print(f"{schema.KEYWORD_INDEX_NAME}: reused")
        deadline = time.monotonic() + WAIT_SECONDS
        while True:
            index = _find(coll)
            status = (index or {}).get("status", "PENDING")
            if index and index.get("queryable"):
                print(f"{schema.KEYWORD_INDEX_NAME}: {status} (queryable)")
                return 0
            if time.monotonic() > deadline:
                print(f"{schema.KEYWORD_INDEX_NAME}: {status}; not queryable after {WAIT_SECONDS}s")
                return 1
            time.sleep(5)
    except PyMongoError as exc:
        code = getattr(exc, "code", None)
        print(f"MongoDB request failed (code {code}). Check connectivity and search-index permission.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
