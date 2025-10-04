"""Utilities for fetching NASA exoplanet tables.

All numeric values are stored in their native mission units; callers are responsible for
normalising downstream. During scaffolding we support a ``--dry-run`` mode that creates
placeholder CSV files so the rest of the stack can be wired without requiring network
access.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import httpx

logger = logging.getLogger(__name__)

TableName = Literal["koi", "k2", "toi"]


@dataclass(frozen=True)
class TableInfo:
    tap_name: str
    description: str


SUPPORTED_TABLES: dict[TableName, TableInfo] = {
    "koi": TableInfo(
        tap_name="koi",
        description="Kepler Objects of Interest cumulative table",
    ),
    "k2": TableInfo(
        tap_name="k2pandc",
        description="K2 planets and planet candidates",
    ),
    "toi": TableInfo(
        tap_name="toi",
        description="TESS Objects of Interest table",
    ),
}


@dataclass(slots=True)
class FetchTableConfig:
    """Configuration for fetching a mission catalog."""

    table: TableName
    output_dir: Path = Path("data/raw")
    overwrite: bool = False
    dry_run: bool = False
    columns: tuple[str, ...] | None = None
    limit: int | None = None
    where: str | None = None


def fetch_table(config: FetchTableConfig) -> Path:
    """Fetch a catalog and return the path to the saved CSV.

    Parameters
    ----------
    config:
        Fetch configuration containing mission table name, output directory, and flags.

    Returns
    -------
    Path
        Location of the CSV file written to disk.
    """

    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{config.table}.csv"
    if output_file.exists() and not config.overwrite:
        logger.info("Table %s already present at %s (overwrite disabled)", config.table, output_file)
        return output_file

    if config.dry_run:
        logger.info("Dry-run: creating placeholder for %s at %s", config.table, output_file)
        placeholder = "id,period_days,depth_ppm,disposition\nplaceholder-001,10.0,500.0,CANDIDATE\n"
        output_file.write_text(placeholder, encoding="utf-8")
        return output_file

    table_info = SUPPORTED_TABLES[config.table]
    query = _build_query(table_info.tap_name, config.columns, config.where, config.limit)
    logger.info("Fetching %s via TAP query: %s", table_info.tap_name, query)

    params = {"query": query, "format": "csv"}
    headers = {
        "User-Agent": "Perilune/0.0.1 (https://github.com/perilune)",
    }

    try:
        with httpx.Client(timeout=httpx.Timeout(120.0)) as client:
            response = client.get(
                "https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
                params=params,
                headers=headers,
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:  # pragma: no cover - network path
        content = exc.response.text[:500]
        raise RuntimeError(
            f"Failed to fetch table {config.table}: {exc.response.status_code} {content}"
        ) from exc
    except httpx.HTTPError as exc:  # pragma: no cover - network path
        raise RuntimeError(f"Network error while fetching table {config.table}: {exc!r}") from exc

    output_file.write_bytes(response.content)
    logger.info("Saved %s rows to %s (approximate)", config.table, output_file)
    return output_file


def _build_query(
    tap_name: str,
    columns: Iterable[str] | None,
    where_clause: str | None,
    limit: int | None,
) -> str:
    select_clause = ",".join(columns) if columns else "*"
    if limit is not None:
        statement = f"SELECT TOP {int(limit)} {select_clause} FROM {tap_name}"
    else:
        statement = f"SELECT {select_clause} FROM {tap_name}"
    if where_clause:
        statement += f" WHERE {where_clause}"
    return statement


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch NASA exoplanet tables")
    parser.add_argument(
        "--table",
        choices=tuple(SUPPORTED_TABLES.keys()),
        required=True,
        help="Table to download",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw"),
        help="Directory to store the CSV (default: data/raw)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create a placeholder file instead of performing a real download",
    )
    parser.add_argument(
        "--columns",
        type=str,
        help="Comma-separated list of columns to select (default: all columns)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of rows returned (appended as FETCH FIRST n ROWS ONLY)",
    )
    parser.add_argument(
        "--where",
        type=str,
        help="Optional SQL WHERE clause (do not include the word WHERE)",
    )
    return parser


def main(argv: list[str] | None = None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    config = FetchTableConfig(
        table=args.table,
        output_dir=args.output,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        columns=tuple(col.strip() for col in args.columns.split(",")) if args.columns else None,
        limit=args.limit,
        where=args.where,
    )
    path = fetch_table(config)
    logger.info("Saved %s table to %s", config.table, path)
    return path


if __name__ == "__main__":  # pragma: no cover - manual invocation entrypoint
    main()
