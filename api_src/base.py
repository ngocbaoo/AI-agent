from __future__ import annotations
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class NormalizedHit:
    """Một cấu trúc dữ liệu đồng nhất cho mọi kết quả tra cứu."""
    source: str
    kind: str
    id: str
    title: str
    url: Optional[str]
    jurisdiction: str
    status: Optional[str]
    filing_date: Optional[str]
    owner: Optional[str]
    classes: Optional[List[str]]
    abstract: Optional[str]

class BaseSource:
    """Lớp cơ sở cho mọi nguồn tra cứu (EUIPO, USPTO, ...)."""
    def search(self, brands: List[str], nice_class: Optional[int] = None) -> List[NormalizedHit]:
        raise NotImplementedError