"""Orchestratore radice del modulo QSCSOP: coordina lettura, elaborazione MAS e persistenza."""

import logging

from qscsop_pipeline.qscsop.interfaces.i_data_sink import IDataSink
from qscsop_pipeline.qscsop.interfaces.i_dataset_adapter import IDatasetAdapter
from qscsop_pipeline.qscsop.mas.interfaces.i_mas_engine import IMASEngine

logger = logging.getLogger(__name__)


class PipelineCoordinator:
    """Coordina il ciclo detect-refactor-validate-review su ogni circuito del dataset di input.

    Componente radice del modulo QSCSOP (sezione 1.4.1), equivalente di QCEPMain per QCEP:
    nessuna interfaccia dedicata, non viene iniettato altrove.
    """

    def __init__(
        self,
        dataset_adapter: IDatasetAdapter,
        mas_engine: IMASEngine,
        data_sink: IDataSink,
    ) -> None:
        self._dataset_adapter = dataset_adapter
        self._mas_engine = mas_engine
        self._data_sink = data_sink

    def run(self) -> None:
        """Elabora ogni circuito del dataset di input e persiste il risultato, uno alla volta.

        Nessun try/except attorno al ciclo: JsonlDatasetAdapter propaga deliberatamente su JSON
        malformato (il dataset di input e' gia' validato a monte da QCEP) e MASEngine garantisce
        sempre un'entita' con stato terminale, mai un'eccezione (sezione 1.4.4) -- aggiungere
        gestione qui sarebbe ridondante e rischierebbe di mascherare un problema reale.
        """
        for index, entity in enumerate(self._dataset_adapter.stream_programs(), start=1):
            logger.info(
                "[%s] Elaborazione circuito %s (%s)...",
                index,
                entity.get_circuit_id(),
                entity.get_dataset_source(),
            )
            result = self._mas_engine.process_entity(entity)
            evaluation = result.get_evaluation()
            logger.info(
                "[%s] Circuito %s completato: stato=%s, iterazioni=%s",
                index,
                result.get_circuit_id(),
                evaluation.get_status(),
                evaluation.get_iteration_count(),
            )
            self._data_sink.save_program(result)
