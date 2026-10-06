from dataclasses import dataclass
from typing import Literal

Pattern = Literal[
    "semantic", "hybrid", "hybrid-reranked", "structured", "decomposition", "hyde"
]


@dataclass(frozen=True)
class Mode:
    pattern: Pattern
    model_id: str


# The one shared registry of course modes. Do not add modes here.
MODES: dict[str, Mode] = {
    m.pattern: m
    for m in (
        Mode("semantic", "rag-semantic"),
        Mode("hybrid", "rag-hybrid"),
        Mode("hybrid-reranked", "rag-hybrid-reranked"),
        Mode("structured", "rag-structured"),
        Mode("decomposition", "rag-decomposition"),
        Mode("hyde", "rag-hyde"),
    )
}

MODES_BY_MODEL_ID: dict[str, Mode] = {m.model_id: m for m in MODES.values()}
