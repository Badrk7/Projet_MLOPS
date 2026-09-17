import yaml
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Config:
    project: str
    data: Dict[str, Any]
    model: Dict[str, Any]
    training: Dict[str, Any]
    app: Dict[str, str]

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(
            project=cfg["project"],
            data=cfg["data"],
            model=cfg["model"],
            training=cfg["training"],
            app=cfg["app"]
        )