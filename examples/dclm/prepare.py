#!/usr/bin/env python3
"""Catalog DCLM Parquet files and prepare one WORLD8 shard set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dclm_example.catalog import build_dclm_catalog
from dclm_example.corpus import (
    DclmCatalog,
    DclmCorpusProducer,
    DclmCursor,
    DclmParquetRowGroupReader,
    DclmTokenization,
    load_tokenizer_encoder,
    publish_dclm_shard_batch,
)

from training_megakernel.runtime import bundle_geometry


def _catalog(args: argparse.Namespace) -> dict[str, Any]:
    return build_dclm_catalog(
        args.output,
        base_directory=args.base_directory,
        ordered_list=args.ordered_list,
        dataset_repo=args.dataset_repo,
        dataset_revision=args.dataset_revision,
    )


def _shards(args: argparse.Namespace) -> dict[str, Any]:
    geometry = bundle_geometry(args.bundle)
    catalog = DclmCatalog.load(args.catalog)
    producer = DclmCorpusProducer(
        catalog=catalog,
        tokenization=DclmTokenization(
            tokenizer_repo=args.tokenizer_repo,
            tokenizer_revision=args.tokenizer_revision,
            eod_token_id=args.eod_token_id,
        ),
        reader=DclmParquetRowGroupReader(args.catalog.parent),
        encode=load_tokenizer_encoder(args.tokenizer_json),
        cursor=None if args.cursor is None else DclmCursor.load(args.cursor),
    )
    result = publish_dclm_shard_batch(
        producer,
        args.output,
        sequence=geometry.sequence,
        steps=args.steps,
        stream_start=args.stream_start,
        embedding_route_records=geometry.route_records,
    )
    return {
        **result,
        "sequence": geometry.sequence,
        "embedding_route_records": geometry.route_records,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Prepare DCLM token shards for resident training."
    )
    commands = result.add_subparsers(dest="command", required=True)

    catalog = commands.add_parser(
        "catalog",
        help="catalog an explicitly ordered set of local DCLM Parquet files",
    )
    catalog.add_argument("ordered_list", type=Path)
    catalog.add_argument("output", type=Path)
    catalog.add_argument("--base-directory", type=Path, required=True)
    catalog.add_argument("--dataset-repo", required=True)
    catalog.add_argument("--dataset-revision", required=True)
    catalog.set_defaults(handler=_catalog)

    shards = commands.add_parser(
        "shards",
        help="materialize one immutable WORLD8 shard set for a training run",
    )
    shards.add_argument("output", type=Path)
    shards.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help=(
            "release bundle the shards are for; its launch_abi.json sets the "
            "tokens per GPU and the embedding-route capacity each window must fit"
        ),
    )
    shards.add_argument("--catalog", type=Path, required=True)
    shards.add_argument("--tokenizer-json", type=Path, required=True)
    shards.add_argument("--tokenizer-repo", required=True)
    shards.add_argument("--tokenizer-revision", required=True)
    shards.add_argument("--eod-token-id", type=int, required=True)
    shards.add_argument("--steps", type=int, required=True)
    shards.add_argument("--stream-start", type=int, default=0)
    shards.add_argument(
        "--cursor",
        type=Path,
        help="continue from a previous shard set's next-cursor.json",
    )
    shards.set_defaults(handler=_shards)
    return result


def main() -> None:
    args = parser().parse_args()
    result = args.handler(args)
    print(json.dumps(result, ensure_ascii=True, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
