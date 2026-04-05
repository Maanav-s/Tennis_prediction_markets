"""Live tennis match + Kalshi market data scraper.

Usage:
    python scrape.py KALSHI_TICKER

Example:
    python scrape.py KXATPCHALLENGERMATCH-26APR04DUCPAC-PAC

The script will:
1. Fetch the Kalshi market details for the given ticker
2. Show live/upcoming matches from API-Tennis and let you pick one
3. Open websockets to both APIs and log data to scraper/data/{ticker}.csv
"""

import argparse
import asyncio
import base64
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

import requests
import websockets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
from dotenv import load_dotenv

SCRAPER_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRAPER_DIR, "data")
load_dotenv(os.path.join(SCRAPER_DIR, ".env"))

KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID")
KALSHI_RSA_PRIVATE_KEY_PATH = os.getenv("KALSHI_RSA_PRIVATE_KEY_PATH")
KALSHI_ENV = os.getenv("KALSHI_ENV", "demo")
API_TENNIS_KEY = os.getenv("API_TENNIS_KEY")

if KALSHI_ENV == "prod":
    KALSHI_REST_BASE = "https://api.elections.kalshi.com/trade-api/v2"
else:
    KALSHI_REST_BASE = "https://demo-api.kalshi.co/trade-api/v2"

API_TENNIS_REST = "https://api.api-tennis.com/tennis/"
API_TENNIS_WS = "wss://wss.api-tennis.com/live"


# ---------------------------------------------------------------------------
# Kalshi auth helpers
# ---------------------------------------------------------------------------

def _load_private_key():
    pem_path = KALSHI_RSA_PRIVATE_KEY_PATH
    if not os.path.isabs(pem_path):
        pem_path = os.path.join(SCRAPER_DIR, pem_path)
    with open(pem_path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def _sign_request(private_key, method: str, path: str, use_pss: bool = False) -> dict:
    ts = str(int(time.time() * 1000))
    msg = (ts + method + path).encode()
    if use_pss:
        pad = asym_padding.PSS(
            mgf=asym_padding.MGF1(hashes.SHA256()),
            salt_length=asym_padding.PSS.MAX_LENGTH,
        )
    else:
        pad = asym_padding.PKCS1v15()
    sig = private_key.sign(msg, pad, hashes.SHA256())
    return {
        "KALSHI-ACCESS-KEY": KALSHI_API_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
    }


def kalshi_get(private_key, path: str, params=None):
    headers = _sign_request(private_key, "GET", path)
    return requests.get(f"{KALSHI_REST_BASE}{path}", headers=headers, params=params)


# ---------------------------------------------------------------------------
# Kalshi market lookup
# ---------------------------------------------------------------------------

def fetch_kalshi_market(private_key, ticker: str) -> dict:
    r = kalshi_get(private_key, f"/markets/{ticker}")
    if r.status_code != 200:
        print(f"Error fetching Kalshi market {ticker}: {r.status_code}")
        print(r.text[:500])
        sys.exit(1)
    return r.json()["market"]


def fetch_kalshi_orderbook(private_key, ticker: str) -> dict:
    r = kalshi_get(private_key, f"/markets/{ticker}/orderbook")
    if r.status_code != 200:
        return {}
    return r.json().get("orderbook_fp", {})


# ---------------------------------------------------------------------------
# API-Tennis match selection
# ---------------------------------------------------------------------------

def fetch_tennis_matches() -> list[dict]:
    """Fetch live + today's upcoming matches from API-Tennis."""
    matches = []

    # Live matches
    r = requests.get(API_TENNIS_REST, params={
        "method": "get_livescore",
        "APIkey": API_TENNIS_KEY,
    })
    if r.status_code == 200:
        data = r.json()
        for m in data.get("result", []):
            m["_source"] = "live"
            matches.append(m)

    # Today's fixtures (upcoming)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    r2 = requests.get(API_TENNIS_REST, params={
        "method": "get_fixtures",
        "APIkey": API_TENNIS_KEY,
        "date_start": today,
        "date_stop": today,
    })
    if r2.status_code == 200:
        data2 = r2.json()
        live_keys = {m["event_key"] for m in matches}
        for m in data2.get("result", []):
            if m["event_key"] not in live_keys:
                m["_source"] = "upcoming"
                matches.append(m)

    return matches


def select_tennis_match(matches: list[dict]) -> dict:
    """Display matches and let the user pick one."""
    if not matches:
        print("No live or upcoming matches found on API-Tennis.")
        sys.exit(1)

    print(f"\n{'#':>3}  {'Status':<10} {'Tournament':<25} {'Match'}")
    print("-" * 80)
    for i, m in enumerate(matches):
        status = m.get("_source", "").upper()
        if m.get("event_status"):
            status = m["event_status"]
        tournament = m.get("tournament_name", "")[:24]
        p1 = m.get("event_first_player", "?")
        p2 = m.get("event_second_player", "?")
        score = m.get("event_final_result", "")
        print(f"{i + 1:>3}  {status:<10} {tournament:<25} {p1} vs {p2}  {score}")

    while True:
        try:
            choice = int(input(f"\nSelect match (1-{len(matches)}): "))
            if 1 <= choice <= len(matches):
                return matches[choice - 1]
        except (ValueError, EOFError):
            pass
        print("Invalid selection, try again.")


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "timestamp",
    "source",          # "kalshi" or "tennis"
    "event_type",      # "orderbook", "trade", "point", "game", "set"
    # Kalshi fields
    "yes_bid", "yes_ask", "yes_bid_size", "yes_ask_size",
    "last_price", "volume",
    # Tennis fields
    "set_score", "game_score", "point_score",
    "server", "point_winner",
    "set_number", "game_number", "point_number",
    "break_point", "set_point", "match_point",
    # Raw JSON for full context
    "raw",
]


class DataLogger:
    def __init__(self, ticker: str):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.path = os.path.join(DATA_DIR, f"{ticker}.csv")
        self.file = open(self.path, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=CSV_FIELDS)
        if self.file.tell() == 0:
            self.writer.writeheader()
        print(f"Logging to {self.path}")

    def log(self, row: dict):
        row.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        # Only keep known fields
        filtered = {k: row.get(k, "") for k in CSV_FIELDS}
        self.writer.writerow(filtered)
        self.file.flush()

    def close(self):
        self.file.close()


# ---------------------------------------------------------------------------
# Kalshi polling
# ---------------------------------------------------------------------------

KALSHI_POLL_INTERVAL = 3  # seconds

async def kalshi_poll(ticker: str, logger: DataLogger, stop_event: asyncio.Event):
    """Poll Kalshi REST API for market and orderbook data."""
    private_key = _load_private_key()
    prev_price = None

    print(f"[Kalshi] Polling every {KALSHI_POLL_INTERVAL}s for {ticker}")

    while not stop_event.is_set():
        try:
            # Market snapshot (best bid/ask, last price, volume)
            market = fetch_kalshi_market(private_key, ticker)
            yes_bid = market.get("yes_bid_dollars", "")
            yes_ask = market.get("yes_ask_dollars", "")
            last_price = market.get("last_price_dollars", "")
            volume = market.get("volume_fp", "")

            logger.log({
                "source": "kalshi",
                "event_type": "market",
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "yes_bid_size": market.get("yes_bid_size_fp", ""),
                "yes_ask_size": market.get("yes_ask_size_fp", ""),
                "last_price": last_price,
                "volume": volume,
                "raw": json.dumps(market),
            })

            if last_price != prev_price:
                print(f"[Kalshi] Yes: {yes_bid}/{yes_ask}  Last: {last_price}  Vol: {volume}")
                prev_price = last_price

        except Exception as e:
            print(f"[Kalshi] Poll error: {e}")

        await asyncio.sleep(KALSHI_POLL_INTERVAL)


# ---------------------------------------------------------------------------
# API-Tennis websocket
# ---------------------------------------------------------------------------

async def tennis_ws(match_key: str, logger: DataLogger, stop_event: asyncio.Event):
    """Connect to API-Tennis websocket and log point-by-point updates."""
    url = f"{API_TENNIS_WS}?APIkey={API_TENNIS_KEY}&match_key={match_key}"

    # Will be set from the first websocket message so we don't replay old points
    prev_point_count = None

    while not stop_event.is_set():
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                print(f"[Tennis] Connected for match {match_key}")

                async for msg in ws:
                    if stop_event.is_set():
                        break
                    try:
                        data = json.loads(msg)
                        matches = data if isinstance(data, list) else [data]

                        for match in matches:
                            event_key = match.get("event_key")
                            if str(event_key) != str(match_key):
                                continue

                            set_score = match.get("event_final_result", "")
                            game_score = match.get("event_game_result", "")
                            server = match.get("event_serve", "")

                            # Extract latest point from pointbypoint
                            pbp = match.get("pointbypoint", [])
                            total_points = sum(
                                len(g.get("points", []))
                                for g in pbp
                            )

                            if prev_point_count is None:
                                # First message — snapshot current state
                                prev_point_count = total_points
                                logger.log({
                                    "source": "tennis",
                                    "event_type": "snapshot",
                                    "set_score": set_score,
                                    "game_score": game_score,
                                    "server": server,
                                    "raw": json.dumps(match),
                                })
                                print(
                                    f"[Tennis] Snapshot: {game_score} "
                                    f"(Set: {set_score}) "
                                    f"Total points so far: {total_points}"
                                )
                                continue

                            if total_points > prev_point_count and pbp:
                                # New point(s) scored — log the latest
                                last_game = pbp[-1]
                                points = last_game.get("points", [])
                                last_point = points[-1] if points else {}

                                logger.log({
                                    "source": "tennis",
                                    "event_type": "point",
                                    "set_score": set_score,
                                    "game_score": game_score,
                                    "point_score": last_point.get("score", ""),
                                    "server": last_game.get("player_served", server),
                                    "point_winner": last_game.get("serve_winner", ""),
                                    "set_number": last_game.get("set_number", ""),
                                    "game_number": last_game.get("number_game", ""),
                                    "point_number": last_point.get("number_point", ""),
                                    "break_point": last_point.get("break_point", ""),
                                    "set_point": last_point.get("set_point", ""),
                                    "match_point": last_point.get("match_point", ""),
                                    "raw": json.dumps(match),
                                })
                                prev_point_count = total_points

                                print(
                                    f"[Tennis] Point: {game_score} "
                                    f"(Set: {set_score}) "
                                    f"Server: {server}"
                                )
                            elif total_points == prev_point_count:
                                # Heartbeat / status update — log game changes
                                status = match.get("event_status", "")
                                logger.log({
                                    "source": "tennis",
                                    "event_type": "status",
                                    "set_score": set_score,
                                    "game_score": game_score,
                                    "server": server,
                                    "raw": json.dumps({"status": status, "score": set_score}),
                                })

                    except json.JSONDecodeError:
                        pass

        except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
            if stop_event.is_set():
                break
            print(f"[Tennis] Connection lost ({e}), reconnecting in 5s...")
            await asyncio.sleep(5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(kalshi_ticker: str, tennis_match_key: str, logger: DataLogger):
    stop_event = asyncio.Event()

    try:
        await asyncio.gather(
            kalshi_poll(kalshi_ticker, logger, stop_event),
            tennis_ws(tennis_match_key, logger, stop_event),
        )
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        logger.close()


def main():
    parser = argparse.ArgumentParser(description="Tennis + Kalshi live data scraper")
    parser.add_argument("ticker", help="Kalshi market ticker")
    args = parser.parse_args()

    ticker = args.ticker
    private_key = _load_private_key()

    # 1. Fetch Kalshi market
    print(f"\nFetching Kalshi market: {ticker}")
    market = fetch_kalshi_market(private_key, ticker)
    print(f"  Title: {market['title']}")
    print(f"  Status: {market['status']}")
    print(f"  Last price: ${market.get('last_price_dollars', '?')}")
    print(f"  Yes bid/ask: ${market.get('yes_bid_dollars', '?')} / ${market.get('yes_ask_dollars', '?')}")

    # 2. Select API-Tennis match
    print("\nFetching live & upcoming tennis matches...")
    matches = fetch_tennis_matches()
    selected = select_tennis_match(matches)
    match_key = selected["event_key"]
    p1 = selected.get("event_first_player", "?")
    p2 = selected.get("event_second_player", "?")
    print(f"\nSelected: {p1} vs {p2} (match_key={match_key})")

    # 3. Start logging
    logger = DataLogger(ticker)
    print("\nStarting websocket streams (Ctrl+C to stop)...\n")

    try:
        asyncio.run(run(ticker, str(match_key), logger))
    except KeyboardInterrupt:
        print("\nStopped.")
        logger.close()


if __name__ == "__main__":
    main()
