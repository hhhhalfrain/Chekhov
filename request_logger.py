from pathlib import Path
import json
from datetime import datetime

LOG_DIR = Path("log")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def log_request_response(request: dict, response: dict):
    """
    保存日志到 log/{timestamp}.json
    timestamp 格式: YYYYMMDDhhmmss (例如 20251029123511)
    直接使用 json.dump(..., default=str) 以简单明了地处理不可序列化对象
    """
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    filepath = LOG_DIR / f"{ts}.json"
    payload = {
        "timestamp": ts,
        "request": request,
        "response": response,
    }
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
