import json
from typing import Any, Dict, List, Mapping

class LabelSchema:
    def __init__(self, schema: Mapping[str, Any]):
        self._raw = dict(schema)
        self.keys: List[str] = list(schema.keys())
        self.titles: Dict[str, str] = {k: schema[k].get("title", k) for k in self.keys}
        self.examples: Dict[str, List[str]] = {k: list(schema[k].get("examples", []) or []) for k in self.keys}

    @staticmethod
    def load(path: str) -> "LabelSchema":
        with open(path, "r", encoding="utf-8") as f:
            return LabelSchema(json.load(f))