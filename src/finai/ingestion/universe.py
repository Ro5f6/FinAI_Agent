"""The set of companies to pre-index, and a check that each can be (1.3.31).

    python -m finai.ingestion.universe reference/sp500-2026-10-20.csv

A universe file is a CSV whose ``#`` lines record where it came from, with
at least a ``ticker`` column, and optionally ``name`` and ``cik`` (a CIK
from the source, to cross-check). The command resolves every ticker as the
queue does (``fetch.resolve_ticker``, 1.3.26) and reports:

* tickers that can't be ingested: ``no_10k`` or ``unknown_ticker``;
* tickers whose 10-Ks are a predecessor's (XOM after its reorganisation);
* tickers whose resolved CIK differs from the file's;
* share classes that resolve to one company (GOOGL and GOOG), which would
  ingest the same 10-K twice under two tickers.

It makes SEC requests only (about two per ticker), no embedding.
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import NamedTuple

import httpx
from edgar.exceptions import EdgarError

from finai.ingestion.fetch import TickerResolution, resolve_ticker


class Constituent(NamedTuple):
    ticker: str
    name: str | None
    cik: int | None  # as the source lists it


def read_universe(path: Path) -> list[Constituent]:
    """The rows of a universe file, skipping its ``#`` source lines."""
    with path.open(newline="") as f:
        rows = csv.DictReader(line for line in f if not line.startswith("#"))
        return [
            Constituent(
                ticker=row["ticker"].strip().upper(),
                name=(row.get("name") or "").strip() or None,
                cik=int(row["cik"]) if (row.get("cik") or "").strip() else None,
            )
            for row in rows
        ]


def read_tickers(path: Path) -> list[str]:
    """Just the tickers, in file order."""
    return [c.ticker for c in read_universe(path)]


class Checked(NamedTuple):
    constituent: Constituent
    resolution: TickerResolution | None
    error: str | None = None

    @property
    def ingestible(self) -> bool:
        return self.resolution is not None and self.resolution.status == "ok"

    @property
    def cik_mismatch(self) -> bool:
        """The resolver's company isn't the one the source lists."""
        listed, r = self.constituent.cik, self.resolution
        if listed is None or r is None or r.cik is None:
            return False
        return listed not in (r.cik, r.current_cik)


def check_universe(
    constituents: Iterable[Constituent],
    resolve: Callable[[str], TickerResolution] | None = None,
    progress: Callable[[int], None] | None = None,
) -> list[Checked]:
    """Resolve each ticker; a network failure is recorded, not raised."""
    resolve = resolve or resolve_ticker
    checked = []
    for i, constituent in enumerate(constituents, 1):
        try:
            checked.append(Checked(constituent, resolve(constituent.ticker)))
        except (httpx.HTTPError, EdgarError) as error:
            checked.append(
                Checked(constituent, None, f"{type(error).__name__}: {error}")
            )
        if progress:
            progress(i)
    return checked


def shared_ciks(checked: Iterable[Checked]) -> dict[int, list[str]]:
    """CIKs that more than one ingestible ticker resolves to."""
    by_cik: dict[int, list[str]] = defaultdict(list)
    for c in checked:
        if c.ingestible:
            by_cik[c.resolution.cik].append(c.constituent.ticker)
    return {cik: tickers for cik, tickers in by_cik.items() if len(tickers) > 1}


def report(checked: list[Checked]) -> str:
    """A plain-text summary listing every ticker that needs attention."""
    ok = [c for c in checked if c.ingestible]
    summary = (
        f"{len(checked)} tickers: {len(ok)} resolve to a 10-K filer, "
        f"{len(checked) - len(ok)} don't."
    )
    lines = [summary]

    def section(title: str, rows: list[str]) -> None:
        lines.append(f"\n{title}: {len(rows)}")
        lines.extend(f"  {row}" for row in rows)

    section(
        "Can't be ingested",
        [
            f"{c.constituent.ticker}: {c.error or c.resolution.status}"
            + (
                f" - {c.resolution.message}"
                if c.resolution and c.resolution.message
                else ""
            )
            for c in checked
            if not c.ingestible
        ],
    )
    section(
        "10-Ks from a predecessor",
        [
            f"{c.constituent.ticker}: {c.resolution.message}"
            for c in ok
            if c.resolution.current_cik
        ],
    )
    section(
        "CIK differs from the file's",
        [
            f"{c.constituent.ticker}: file {c.constituent.cik}, resolved {c.resolution.cik}"
            f" ({c.resolution.name})"
            for c in checked
            if c.cik_mismatch
        ],
    )
    section(
        "One company, several tickers",
        [
            f"CIK {cik}: {', '.join(tickers)}"
            for cik, tickers in shared_ciks(checked).items()
        ],
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m finai.ingestion.universe",
        description="Check that every ticker in a universe file resolves to a 10-K filer.",
    )
    parser.add_argument(
        "path", type=Path, help="universe CSV, e.g. reference/sp500-2026-10-20.csv"
    )
    args = parser.parse_args(argv)

    constituents = read_universe(args.path)
    started = time.monotonic()

    def progress(i: int) -> None:
        if i % 50 == 0 or i == len(constituents):
            print(f"  resolved {i}/{len(constituents)}", file=sys.stderr, flush=True)

    checked = check_universe(constituents, progress=progress)
    print(report(checked))
    print(f"\n{time.monotonic() - started:.0f} s", file=sys.stderr)
    return 0 if all(c.ingestible for c in checked) else 1


if __name__ == "__main__":
    sys.exit(main())
