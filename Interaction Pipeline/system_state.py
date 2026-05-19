"""Online/offline state helpers for the interaction pipeline."""

import os
import socket


def is_online(timeout: float = 1.5) -> bool:
    """Return True when outbound connectivity appears available.

    Set KNIGHTRO_FORCE_OFFLINE=true to force offline behavior during tests.
    """
    forced = os.getenv("KNIGHTRO_FORCE_OFFLINE", "").strip().lower()
    if forced in {"1", "true", "yes"}:
        return False

    probes = [("8.8.8.8", 53), ("1.1.1.1", 53)]
    for host, port in probes:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            continue
    return False


if __name__ == "__main__":
    print("ONLINE" if is_online() else "OFFLINE")
