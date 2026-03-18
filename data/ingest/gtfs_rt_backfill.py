"""
gtfs_rt_backfill.py – Reads historical MTA GTFS-Realtime protobuf archives
from S3 (s3://mta-gtfs-rt-archives) and backfills the ``delay_index`` table
in TimescaleDB.

S3 key layout assumed::

    s3://mta-gtfs-rt-archives/{YYYY}/{MM}/{DD}/{HH}/gtfs_rt.pb

Usage::

    python -m data.ingest.gtfs_rt_backfill \\
        --start 2023-01-01 \\
        --end   2024-06-30

Environment variables:
    TIMESCALE_DSN   – PostgreSQL DSN (default: local dev).
    AWS_PROFILE     – Optional named AWS profile.
    AWS_REGION      – AWS region (default: us-east-1).
    BACKFILL_BUCKET – S3 bucket name (default: mta-gtfs-rt-archives).
    BACKFILL_PREFIX – Key prefix within the bucket (default: "").
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import UTC, date, datetime, timedelta

import boto3
from botocore.exceptions import ClientError
from google.protobuf.message import DecodeError
from google.transit import gtfs_realtime_pb2

from data.ingest.gtfs_rt import _compute_delay_index, _upsert_rows

logger = logging.getLogger(__name__)

_BUCKET = os.getenv("BACKFILL_BUCKET", "mta-gtfs-rt-archives")
_PREFIX = os.getenv("BACKFILL_PREFIX", "")
_REGION = os.getenv("AWS_REGION", "us-east-1")

# Key template matching the MTA archive layout.
_KEY_TEMPLATE = "{prefix}{year:04d}/{month:02d}/{day:02d}/{hour:02d}/gtfs_rt.pb"


def _date_range(start: date, end: date) -> list[date]:
    """Return every date from *start* up to and including *end*."""
    result: list[date] = []
    current = start
    while current <= end:
        result.append(current)
        current += timedelta(days=1)
    return result


def _s3_key(d: date, hour: int) -> str:
    """Build the S3 key for a given date and hour."""
    prefix = (_PREFIX.rstrip("/") + "/") if _PREFIX else ""
    return _KEY_TEMPLATE.format(
        prefix=prefix,
        year=d.year,
        month=d.month,
        day=d.day,
        hour=hour,
    )


def _fetch_archive(s3_client, key: str) -> gtfs_realtime_pb2.FeedMessage | None:
    """Download a single protobuf archive from S3 and parse it.

    Returns *None* if the object does not exist or cannot be parsed.
    """
    try:
        response = s3_client.get_object(Bucket=_BUCKET, Key=key)
        data = response["Body"].read()
    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        if error_code in {"NoSuchKey", "404"}:
            logger.debug("Key not found: s3://%s/%s – skipping.", _BUCKET, key)
        else:
            logger.warning("S3 error for key %s: %s", key, exc)
        return None

    feed = gtfs_realtime_pb2.FeedMessage()
    try:
        feed.ParseFromString(data)
    except DecodeError:
        logger.warning("Failed to parse protobuf at s3://%s/%s", _BUCKET, key)
        return None

    # If the feed header has no timestamp, inject a synthetic one derived from
    # the key path so that _compute_delay_index can bucket by hour correctly.
    if feed.header.timestamp == 0:
        # key pattern: .../YYYY/MM/DD/HH/gtfs_rt.pb
        parts = key.rstrip("/").split("/")
        try:
            hh = int(parts[-2])
            dd = int(parts[-3])
            mm = int(parts[-4])
            yyyy = int(parts[-5])
            ts = int(
                datetime(yyyy, mm, dd, hh, 0, 0, tzinfo=UTC).timestamp()
            )
            feed.header.timestamp = ts
        except (ValueError, IndexError):
            logger.warning(
                "Cannot derive timestamp from key path %r – skipping file.", key
            )
            return None

    return feed


def backfill(
    start: date,
    end: date,
    *,
    aws_profile: str | None = None,
    dry_run: bool = False,
) -> int:
    """Backfill ``delay_index`` rows from S3 archives.

    Args:
        start:       First date (inclusive) to backfill.
        end:         Last date (inclusive) to backfill.
        aws_profile: Optional named AWS credential profile.
        dry_run:     If *True*, parse and log but do not write to the DB.

    Returns:
        Total number of rows upserted (or that would have been upserted in
        dry-run mode).
    """
    session = boto3.Session(profile_name=aws_profile, region_name=_REGION)
    s3 = session.client("s3")

    total_rows = 0
    for d in _date_range(start, end):
        for hour in range(24):
            key = _s3_key(d, hour)
            feed = _fetch_archive(s3, key)
            if feed is None:
                continue

            rows = _compute_delay_index(feed)
            if not rows:
                logger.debug("No delay rows for %s/%02d", d, hour)
                continue

            logger.info(
                "s3://%s/%s → %d rows (dry_run=%s)",
                _BUCKET,
                key,
                len(rows),
                dry_run,
            )
            if not dry_run:
                _upsert_rows(rows)
            total_rows += len(rows)

    return total_rows


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill delay_index from MTA GTFS-RT S3 archives.",
    )
    parser.add_argument(
        "--start",
        required=True,
        type=date.fromisoformat,
        help="Start date in ISO format (YYYY-MM-DD), inclusive.",
    )
    parser.add_argument(
        "--end",
        default=date.today(),
        type=date.fromisoformat,
        help="End date in ISO format (YYYY-MM-DD), inclusive. Defaults to today.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Named AWS credential profile to use.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and log rows without writing to the database.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s – %(message)s",
    )
    args = _parse_args(argv)

    if args.start > args.end:
        raise SystemExit("--start must be on or before --end")

    logger.info(
        "Backfilling delay_index from %s to %s (dry_run=%s) …",
        args.start,
        args.end,
        args.dry_run,
    )
    total = backfill(args.start, args.end, aws_profile=args.profile, dry_run=args.dry_run)
    logger.info("Backfill complete – %d rows processed.", total)


if __name__ == "__main__":
    main()
