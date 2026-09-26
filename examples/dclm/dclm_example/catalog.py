"""Build a DCLM catalog from one explicit local file order."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .corpus import DCLM_CATALOG_SCHEMA

DCLM_ORDERED_LOCAL_FILES_SCHEMA = "training_megakernel_dclm_file_list_v1"


def build_dclm_catalog(
    output: str | Path,
    *,
    base_directory: str | Path,
    ordered_list: str | Path,
    dataset_repo: str,
    dataset_revision: str,
) -> dict[str, Any]:
    """Inspect and write a catalog in exactly the caller-supplied file order."""

    try:
        from pyarrow import parquet
    except ImportError as error:
        raise RuntimeError("DCLM catalog construction requires the 'dclm' extra") from error

    base = Path(base_directory)
    destination = Path(output)
    if destination.parent.resolve() != base.resolve():
        raise ValueError("catalog output must live directly in its Parquet base directory")
    if destination.exists():
        raise FileExistsError(destination)
    listing = json.loads(Path(ordered_list).read_bytes())
    if listing.get("schema") != DCLM_ORDERED_LOCAL_FILES_SCHEMA:
        raise ValueError("unexpected ordered Parquet list schema")
    files: list[dict[str, Any]] = []
    for relative in listing["files"]:
        path = base / relative
        metadata = parquet.ParquetFile(path).metadata
        files.append(
            {
                "bytes": path.stat().st_size,
                "path": relative,
                "row_group_rows": [
                    metadata.row_group(index).num_rows
                    for index in range(metadata.num_row_groups)
                ],
            }
        )
    payload = json.dumps(
        {
            "dataset_repo": dataset_repo,
            "dataset_revision": dataset_revision,
            "files": files,
            "schema": DCLM_CATALOG_SCHEMA,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, destination)
    return {
        "catalog": str(destination),
        "file_count": len(files),
        "row_count": sum(sum(row["row_group_rows"]) for row in files),
        "row_group_count": sum(len(row["row_group_rows"]) for row in files),
    }


__all__ = ("DCLM_ORDERED_LOCAL_FILES_SCHEMA", "build_dclm_catalog")
