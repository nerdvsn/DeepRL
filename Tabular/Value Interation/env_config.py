from dataclasses import dataclass
from typing import Optional, List

@dataclass
class EnvConfig:
    desc: Optional[List[str]] = None       # eigene Map, z.B. ["SFFF","FHFH","FFFH","HFFG"]
    map_name: Optional[str] = None         # "4x4" oder "8x8" (nicht gemeinsam mit desc nutzen)
    is_slippery: bool = False
    seed: int = 12345
