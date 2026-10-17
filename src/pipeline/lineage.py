import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LineageGraph:
    run_id: str
    datasets: dict[str, dict] = field(default_factory=dict)
    steps: list[dict] = field(default_factory=list)

    def dataset(self, name: str, stage: str, rows: int | None = None, fingerprint: str | None = None) -> str:
        self.datasets[name] = {"stage": stage, "rows": rows, "fingerprint": fingerprint}
        return name

    def step(self, name: str, inputs: list[str], outputs: list[str]) -> None:
        unknown = [d for d in inputs + outputs if d not in self.datasets]
        if unknown:
            raise KeyError(f"Lineage step '{name}' references unregistered datasets: {unknown}")
        self.steps.append({"step": name, "inputs": inputs, "outputs": outputs})

    def write(self, path: Path) -> Path:
        path.write_text(json.dumps({"run_id": self.run_id, "datasets": self.datasets, "steps": self.steps}, indent=2))
        return path
