"""Implementazione concreta della lettura JSON Lines del dataset prodotto da QCEP."""

import json
from collections.abc import Generator
from pathlib import Path

from qscsop_pipeline.qscsop.entities.circuit_version import CircuitVersion
from qscsop_pipeline.qscsop.entities.smell_metrics import SmellMetrics
from qscsop_pipeline.qscsop.entities.quantum_program_entity import QuantumProgramEntity
from qscsop_pipeline.qscsop.interfaces.i_dataset_adapter import IDatasetAdapter


class JsonlDatasetAdapter(IDatasetAdapter):
    """Legge un file JSON Lines (formato Listing 1.2) e lo ricostruisce come entità di dominio.

    Il formato letto e' quello prodotto da QuantumMetricsService: sorgente piu' misura QSMELL.
    I due lati non condividono codice -- lo schema e' il contratto implicito fra QCEP e QSCSOP --
    quindi vanno tenuti allineati a mano.
    """

    def __init__(self, filepath: str) -> None:
        self._filepath = Path(filepath)

    def stream_programs(self) -> Generator[QuantumProgramEntity, None, None]:
        """Genera un QuantumProgramEntity per riga, leggendo il file senza caricarlo in memoria."""
        with self._filepath.open("r", encoding="utf-8") as input_file:
            for line in input_file:
                record = json.loads(line)
                yield self._to_entity(record)

    @staticmethod
    def _to_entity(record: dict) -> QuantumProgramEntity:
        """Ricostruisce un QuantumProgramEntity a partire da un record grezzo già validato.

        Un errore qui (chiave mancante, JSON malformato) indica un dataset di input corrotto
        o incompatibile: non viene ingoiato, a differenza di QCEP, perché a questo stadio il
        file e' gia' stato prodotto e validato da QCEP stesso.
        """
        baseline_record = record["baseline"]
        smell_record = baseline_record["smellMetrics"]
        # longCircuit non viene letto: e' derivato da SmellMetrics come prodotto dei due fattori.
        # Ricostruirlo dal file significherebbe poter caricare un'entita' in cui il prodotto non
        # corrisponde ai suoi fattori, cioe' rimettere dentro l'incoerenza che la property evita.
        baseline = CircuitVersion(
            source_code=baseline_record["sourceCode"],
            smell_metrics=SmellMetrics(
                max_ops_per_qubit=smell_record["maxOpsPerQubit"],
                max_parallel_ops=smell_record["maxParallelOps"],
                idle_qubits=smell_record["idleQubits"],
            ),
        )
        return QuantumProgramEntity(
            circuit_id=record["circuitId"],
            dataset_source=record["datasetSource"],
            baseline=baseline,
        )
