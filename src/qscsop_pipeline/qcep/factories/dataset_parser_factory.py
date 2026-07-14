"""Factory che istanzia il parser concreto appropriato per un dataset nominato."""

from pathlib import Path

from qscsop_pipeline.qcep.interfaces.i_dataset_parser import IDatasetParser
from qscsop_pipeline.qcep.parsers.bugs4q_dataset_parser import Bugs4QDatasetParser
from qscsop_pipeline.qcep.parsers.the_smelly_eight_dataset_parser import (
    TheSmellyEightDatasetParser,
)

# src/qscsop_pipeline/qcep/factories/dataset_parser_factory.py -> risale alla root del progetto,
# cosi' i path dei dataset restano corretti indipendentemente dalla working directory di lancio.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DATA_RAW = _PROJECT_ROOT / "data" / "raw"


class DatasetParserFactory:
    """Astrae la scelta e l'istanziazione del parser corretto per un dataset."""

    def get_parser(self, dataset_name: str) -> IDatasetParser:
        """Ritorna il parser concreto associato a dataset_name (case-insensitive)."""
        normalized_name = dataset_name.lower()

        if normalized_name == "bugs4q":
            return Bugs4QDatasetParser(dataset_path=str(_DATA_RAW / "bugs4q"))
        if normalized_name == "thesmellyeight":
            return TheSmellyEightDatasetParser(dataset_path=str(_DATA_RAW / "thesmellyeight"))

        raise ValueError(
            f"Dataset '{dataset_name}' non riconosciuto. Valori validi: bugs4q, thesmellyeight."
        )
