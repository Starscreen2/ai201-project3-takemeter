#!/usr/bin/env python3
"""Fetch public posts and comments from a subreddit for TakeMeter labeling."""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path

import requests

USER_AGENT = "TakeMeter/1.0 (CodePath AI201 educational project)"
MIN_TEXT_LEN = 40
MAX_TEXT_LEN = 512
ARCTIC_SHIFT = "https://arctic-shift.photon-reddit.com/api"
PULLPUSH = "https://api.pullpush.io/reddit/search"


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text


def fetch_reddit_json(
    subreddit: str,
    listing: str,
    limit: int,
    session: requests.Session,
) -> list[dict]:
    url = f"https://www.reddit.com/r/{subreddit}/{listing}.json?limit={limit}"
    response = session.get(url, timeout=30)
    if response.status_code == 403:
        return []
    response.raise_for_status()
    children = response.json().get("data", {}).get("children", [])
    rows: list[dict] = []
    for child in children:
        post = child.get("data", {})
        post_id = post.get("id")
        if not post_id:
            continue
        title = clean_text(post.get("title", ""))
        body = clean_text(post.get("selftext", ""))
        post_text = clean_text(f"{title}. {body}" if body else title)
        if MIN_TEXT_LEN <= len(post_text) <= MAX_TEXT_LEN:
            rows.append(
                {
                    "text": post_text,
                    "source": f"reddit_post:{listing}",
                    "reddit_id": post_id,
                }
            )
    return rows


def fetch_arctic_posts(
    subreddit: str,
    limit: int,
    session: requests.Session,
) -> list[dict]:
    rows: list[dict] = []
    after: int | None = None
    while len(rows) < limit:
        params: dict[str, str | int] = {
            "subreddit": subreddit,
            "limit": min(100, limit - len(rows)),
            "sort": "desc",
        }
        if after is not None:
            params["after"] = after
        response = session.get(f"{ARCTIC_SHIFT}/posts/search", params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", [])
        if not data:
            break
        for post in data:
            post_id = post.get("id")
            if not post_id:
                continue
            title = clean_text(post.get("title", ""))
            body = clean_text(post.get("selftext", ""))
            post_text = clean_text(f"{title}. {body}" if body else title)
            if MIN_TEXT_LEN <= len(post_text) <= MAX_TEXT_LEN:
                rows.append(
                    {
                        "text": post_text,
                        "source": "arctic_post",
                        "reddit_id": post_id,
                    }
                )
        after = data[-1].get("created_utc")
        if after is None:
            break
        time.sleep(0.5)
    return rows


def fetch_pullpush_comments(
    subreddit: str,
    limit: int,
    session: requests.Session,
) -> list[dict]:
    rows: list[dict] = []
    before: int | None = None
    while len(rows) < limit:
        params: dict[str, str | int] = {
            "subreddit": subreddit,
            "size": min(100, limit - len(rows)),
            "sort": "desc",
            "sort_type": "created_utc",
        }
        if before is not None:
            params["before"] = before
        response = session.get(f"{PULLPUSH}/comment/", params=params, timeout=30)
        response.raise_for_status()
        data = response.json().get("data", [])
        if not data:
            break
        for comment in data:
            comment_id = comment.get("id")
            body = clean_text(comment.get("body", ""))
            if not comment_id or body in {"[deleted]", "[removed]"}:
                continue
            if MIN_TEXT_LEN <= len(body) <= MAX_TEXT_LEN:
                rows.append(
                    {
                        "text": body,
                        "source": "pullpush_comment",
                        "reddit_id": comment_id,
                    }
                )
        before = data[-1].get("created_utc")
        if before is None:
            break
        time.sleep(0.5)
    return rows


def dedupe_rows(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for row in rows:
        reddit_id = row["reddit_id"]
        if reddit_id in seen:
            continue
        seen.add(reddit_id)
        unique.append(row)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Reddit text for TakeMeter")
    parser.add_argument("--subreddit", default="leagueoflegends")
    parser.add_argument("--target", type=int, default=250)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw_posts.csv",
    )
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    rows: list[dict] = []
    for listing in ("hot", "new", "top"):
        rows.extend(fetch_reddit_json(args.subreddit, listing, 50, session))

    if len(rows) < args.target:
        print("Reddit JSON blocked or sparse; using Arctic Shift + PullPush archives.")
        post_target = args.target // 2
        comment_target = args.target - post_target
        rows.extend(fetch_arctic_posts(args.subreddit, post_target + 40, session))
        rows.extend(fetch_pullpush_comments(args.subreddit, comment_target + 40, session))

    rows = dedupe_rows(rows)[: args.target]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "source", "reddit_id"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
