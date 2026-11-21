"""A read-only probe of a Coherent OBIS Laser Box: it sends only queries, never a setting, and
prints every raw reply line, so the protocol can be checked against docs/obis_laser_box.md.

Close Coherent Connection first (it holds the port), then run:
    python docs/obis_probe.py COM8 > obis_probe_COM8.txt
"""

from __future__ import annotations

import sys
import time

import serial

# Every reply is read until the line has been quiet this long, so the transcript shows the framing
# as it is (value, OK, ERR, prompt) instead of as a parser expects it.
QUIET_S = 0.3
# A single exchange is cut off after this long, in case a reply never goes quiet.
EXCHANGE_LIMIT_S = 3.0

# Unnumbered queries, to learn which addressee answers when no channel is given.
UNNUMBERED = [
    "*IDN?",
    "SYST:INF:MOD?",
    "SYST:INF:WAV?",
    "SYST:STAT?",
    "SOUR:AM:SOUR?",
]

# Asked of channel 0 (the box) and of each bay 1-5; ``{n}`` is the channel.
PER_CHANNEL = [
    "*IDN{n}?",
    "SYST{n}:INF:MOD?",
    "SYSTem{n}:INFormation:MODel?",
    "SYST{n}:INF:SNUM?",
    "SYST{n}:INF:FVER?",
    "SYST{n}:INF:WAV?",
    "SYST{n}:INF:POW?",
    "SYST{n}:INF:TYP?",
    "SYST{n}:STAT?",
    "SYST{n}:FAUL?",
    "SYST{n}:COMM:HAND?",
    "SYST{n}:COMM:PROM?",
    "SYST{n}:CDRH?",
    "SYST{n}:AUT?",
    "SYST{n}:LOCK?",
    "SYST{n}:HOUR?",
    "SYST{n}:DIOD:HOUR?",
    "SYST{n}:ERR:COUN?",
    "SOUR{n}:AM:STAT?",
    "SOUR{n}:AM:SOUR?",
    "SOUR{n}:POW:LEV:IMM:AMPL?",
    "SOUR{n}:POW:LEV?",
    "SOUR{n}:POW:NOM?",
    "SOUR{n}:POW:LIM:LOW?",
    "SOUR{n}:POW:LIM:HIGH?",
    "SOUR{n}:POW:CURR?",
    "SOUR{n}:TEMP:BAS?",
    "SOUR{n}:TEMP:DIOD?",
    "SOUR{n}:TEMP:INT?",
]


def ask(port: serial.Serial, query: str) -> tuple[list[bytes], float]:
    """Send one query and collect every line until the port goes quiet, with the time to the
    first line."""
    if not query.endswith("?"):
        raise ValueError(f"the probe sends only queries, not {query!r}")
    port.write(query.encode("ascii") + b"\r")
    sent = time.monotonic()
    first = 0.0
    lines: list[bytes] = []
    while time.monotonic() - sent < EXCHANGE_LIMIT_S:
        line = port.readline()
        if not line:
            break
        if not lines:
            first = time.monotonic() - sent
        lines.append(line)
    return lines, first


def probe(port: serial.Serial, queries: list[str]) -> None:
    for query in queries:
        lines, first = ask(port, query)
        print(f"{query:<34} {first * 1000:6.1f} ms  {lines!r}", flush=True)


def main() -> None:
    port_name = sys.argv[1]
    with serial.Serial(port_name, baudrate=115200, timeout=QUIET_S) as port:
        # Whatever was left on the line before the probe started.
        time.sleep(QUIET_S)
        print(f"# {port_name}: stale input {port.read(port.in_waiting)!r}")
        print("# unnumbered")
        probe(port, UNNUMBERED)
        for n in range(6):
            print(f"# channel {n}")
            probe(port, [query.format(n=n) for query in PER_CHANNEL])


if __name__ == "__main__":
    main()
