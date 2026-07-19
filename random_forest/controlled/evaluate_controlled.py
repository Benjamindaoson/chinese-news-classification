from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "reports" / "random_forest" / "controlled" / "controlled_metrics.json"


def main() -> int:
    if not METRICS.exists():
        raise FileNotFoundError(f"missing controlled metrics: {METRICS}")
    data = json.loads(METRICS.read_text(encoding="utf-8"))
    print(json.dumps({
        "leaky_test_macro_f1": data["leaky"]["test"]["macro_f1"],
        "clean_test_macro_f1": data["clean"]["test"]["macro_f1"],
        "rf_params_identical": data["rf_params_identical"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
