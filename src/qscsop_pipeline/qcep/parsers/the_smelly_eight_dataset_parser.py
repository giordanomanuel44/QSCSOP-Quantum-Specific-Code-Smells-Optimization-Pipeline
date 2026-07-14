"""Parser del dataset TheSmellyEight: estrae solo le versioni smelly, ignora le fixed."""

from collections.abc import Generator
from pathlib import Path

from qscsop_pipeline.qcep.interfaces.i_dataset_parser import IDatasetParser

_SMELLY_STEM_SUFFIX = "-smelly"


class TheSmellyEightDatasetParser(IDatasetParser):
    """Estrae record grezzi dai file *-smelly.py annidati sotto le sottocartelle smell."""

    _DATASET_SOURCE = "TheSmellyEight"

    def __init__(self, dataset_path: str = "data/raw/thesmellyeight") -> None:
        self._dataset_path = Path(dataset_path)

    def extract_circuits(self) -> Generator[dict, None, None]:
        """Genera un record grezzo per ciascun file *-smelly.py trovato ricorsivamente."""
        for source_file in self._locate_source_files():
            yield {
                "circuitId": self._generate_circuit_id(source_file),
                "datasetSource": self._DATASET_SOURCE,
                "sourceCode": self._read_source_code(source_file),
            }

    def _locate_source_files(self) -> Generator[Path, None, None]:
        """Percorre ricorsivamente le sottocartelle smell, filtrando per suffisso -smelly.py.

        L'estensione .py e' confrontata case-insensitive, lo stem -smelly con match esatto.
        """
        for candidate in self._dataset_path.rglob("*"):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() != ".py":
                continue
            if candidate.stem.endswith(_SMELLY_STEM_SUFFIX):
                yield candidate

    @staticmethod
    def _read_source_code(source_file: Path) -> str:
        """Legge il contenuto testuale del file sorgente."""
        return source_file.read_text(encoding="utf-8")

    @staticmethod
    def _generate_circuit_id(source_file: Path) -> str:
        """Deriva il circuitId dallo stem del filename, per restare tracciabile alla fonte."""
        return source_file.stem
