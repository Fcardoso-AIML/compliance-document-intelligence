import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict


def log_event(log_path: Path, event_type: str, payload: Dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "ts": datetime.now(UTC).isoformat(),
        "event_type": event_type,
        "payload": payload,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
