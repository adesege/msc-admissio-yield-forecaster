from dataclasses import dataclass
from pathlib import Path

import yaml


class AccessDenied(PermissionError):
    pass


@dataclass(frozen=True)
class DatasetPolicy:
    name: str
    classification: str
    retention: dict
    access: dict[str, frozenset[str]]


@dataclass(frozen=True)
class GovernancePolicy:
    roles: dict[str, str]
    datasets: dict[str, DatasetPolicy]

    @classmethod
    def from_yaml(cls, path: Path) -> "GovernancePolicy":
        document = yaml.safe_load(path.read_text())
        datasets = {
            name: DatasetPolicy(
                name=name,
                classification=spec["classification"],
                retention=spec["retention"],
                access={role: frozenset(actions) for role, actions in spec["access"].items()},
            )
            for name, spec in document["datasets"].items()
        }
        unknown_roles = {role for spec in datasets.values() for role in spec.access} - set(document["roles"])
        if unknown_roles:
            raise ValueError(f"Policy grants access to undeclared roles: {sorted(unknown_roles)}")
        return cls(roles=document["roles"], datasets=datasets)

    def dataset(self, name: str) -> DatasetPolicy:
        if name not in self.datasets:
            raise AccessDenied(f"Dataset '{name}' is not registered in the governance policy")
        return self.datasets[name]

    def is_allowed(self, role: str, dataset: str, action: str) -> bool:
        return action in self.dataset(dataset).access.get(role, frozenset())

    def authorize(self, role: str, dataset: str, action: str) -> None:
        if not self.is_allowed(role, dataset, action):
            raise AccessDenied(f"Role '{role}' may not {action} dataset '{dataset}'")
