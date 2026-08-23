from unittest.mock import Mock

import pytest

from qscsop_pipeline.qscsop.interfaces.i_data_sink import IDataSink
from qscsop_pipeline.qscsop.interfaces.i_dataset_adapter import IDatasetAdapter
from qscsop_pipeline.qscsop.mas.interfaces.i_mas_engine import IMASEngine
from qscsop_pipeline.qscsop.coordinator.pipeline_coordinator import PipelineCoordinator


def _make_collaborators() -> dict:
    return {
        "dataset_adapter": Mock(spec=IDatasetAdapter),
        "mas_engine": Mock(spec=IMASEngine),
        "data_sink": Mock(spec=IDataSink),
    }


@pytest.mark.unit
def test_run_calls_process_entity_once_per_entity_in_order() -> None:
    collaborators = _make_collaborators()
    entities = [Mock(name="entity1"), Mock(name="entity2"), Mock(name="entity3")]
    collaborators["dataset_adapter"].stream_programs.return_value = iter(entities)
    collaborators["mas_engine"].process_entity.side_effect = lambda entity: entity
    coordinator = PipelineCoordinator(**collaborators)

    coordinator.run()

    assert collaborators["mas_engine"].process_entity.call_count == 3
    processed_in_order = [
        call.args[0] for call in collaborators["mas_engine"].process_entity.call_args_list
    ]
    assert processed_in_order == entities


@pytest.mark.unit
def test_run_saves_the_object_returned_by_process_entity_not_the_original() -> None:
    collaborators = _make_collaborators()
    original_entity = Mock(name="original")
    mutated_result = Mock(name="mutated_result")
    collaborators["dataset_adapter"].stream_programs.return_value = iter([original_entity])
    collaborators["mas_engine"].process_entity.return_value = mutated_result
    coordinator = PipelineCoordinator(**collaborators)

    coordinator.run()

    collaborators["data_sink"].save_program.assert_called_once_with(mutated_result)
    # Conferma esplicita: non e' l'entita' originale ad essere passata al sink.
    saved_argument = collaborators["data_sink"].save_program.call_args.args[0]
    assert saved_argument is mutated_result
    assert saved_argument is not original_entity


@pytest.mark.unit
def test_run_with_empty_stream_calls_nothing_and_raises_nothing() -> None:
    collaborators = _make_collaborators()
    collaborators["dataset_adapter"].stream_programs.return_value = iter([])
    coordinator = PipelineCoordinator(**collaborators)

    coordinator.run()

    collaborators["mas_engine"].process_entity.assert_not_called()
    collaborators["data_sink"].save_program.assert_not_called()


@pytest.mark.unit
def test_run_consumes_stream_programs_lazily_one_entity_at_a_time() -> None:
    """Conferma il vincolo O(1): stream_programs() non viene mai materializzato in una lista.

    Se PipelineCoordinator convertisse il generator in lista prima di elaborare (es.
    list(self._dataset_adapter.stream_programs())), tutti gli yield comparirebbero consecutivi
    in call_order PRIMA di qualunque process/save. L'ordine atteso qui intercala invece
    esplicitamente yield, process e save per ciascuna entita', una alla volta.
    """
    collaborators = _make_collaborators()
    entities = [Mock(name=f"entity{i}") for i in range(3)]
    call_order: list[str] = []

    def stream_programs():
        for index, entity in enumerate(entities):
            call_order.append(f"yield_{index}")
            yield entity

    def process_entity(entity):
        index = entities.index(entity)
        call_order.append(f"process_{index}")
        return entity

    def save_program(entity):
        index = entities.index(entity)
        call_order.append(f"save_{index}")

    collaborators["dataset_adapter"].stream_programs.side_effect = stream_programs
    collaborators["mas_engine"].process_entity.side_effect = process_entity
    collaborators["data_sink"].save_program.side_effect = save_program
    coordinator = PipelineCoordinator(**collaborators)

    coordinator.run()

    assert call_order == [
        "yield_0", "process_0", "save_0",
        "yield_1", "process_1", "save_1",
        "yield_2", "process_2", "save_2",
    ]
