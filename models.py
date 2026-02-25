from dataclasses import dataclass

@dataclass
class IngestDocument:
    type : str
    path : str
    name : str
    text : str