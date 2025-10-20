import json
from pathlib import Path

from salience_os_seed.runtime.smoke_runtime import run_smoke


def test_smoke_runtime_runs(tmp_path: Path):
    output = tmp_path / "metrics.json"
    run_smoke(times=2, output=str(output))
    assert output.exists()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert len(data) == 2
    for entry in data:
        assert "step" in entry
        assert "maintenance" in entry
        assert "experiments" in entry
