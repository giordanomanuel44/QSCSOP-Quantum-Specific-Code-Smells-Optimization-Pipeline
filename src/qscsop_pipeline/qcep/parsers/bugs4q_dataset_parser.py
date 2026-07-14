"""Parser del dataset Bugs4Q: legge le versioni fixed dei bug gia' filtrate su disco."""

from collections.abc import Generator
from pathlib import Path

from qscsop_pipeline.qcep.interfaces.i_dataset_parser import IDatasetParser


class Bugs4QDatasetParser(IDatasetParser):
    """Estrae record grezzi dai file .py piatti di data/raw/bugs4q."""

    _DATASET_SOURCE = "Bugs4Q"

    def __init__(self, dataset_path: str = "data/raw/bugs4q") -> None:
        self._dataset_path = Path(dataset_path)

    def extract_circuits(self) -> Generator[dict, None, None]:
        """Genera un record grezzo per ciascun file .py nella cartella del dataset."""
        for source_file in self._locate_source_files():
            yield {
                "circuitId": self._generate_circuit_id(source_file),
                "datasetSource": self._DATASET_SOURCE,
                "sourceCode": self._read_source_code(source_file),
            }

    def _locate_source_files(self) -> Generator[Path, None, None]:
        """Elenca i file .py nella cartella del dataset, senza ricorsione."""
        yield from self._dataset_path.glob("*.py")

    @staticmethod
    def _read_source_code(source_file: Path) -> str:
        """Legge il contenuto testuale del file sorgente."""
        return source_file.read_text(encoding="utf-8")

    @staticmethod
    def _generate_circuit_id(source_file: Path) -> str:
        """Deriva il circuitId dallo stem del filename, per restare tracciabile alla fonte."""
        return source_file.stem
