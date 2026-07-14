import pytest

from qscsop_pipeline.qcep.factories.dataset_parser_factory import DatasetParserFactory
from qscsop_pipeline.qcep.parsers.bugs4q_dataset_parser import Bugs4QDatasetParser
from qscsop_pipeline.qcep.parsers.the_smelly_eight_dataset_parser import (
    TheSmellyEightDatasetParser,
)


@pytest.fixture
def factory() -> DatasetParserFactory:
    return DatasetParserFactory()


@pytest.mark.unit
def test_get_parser_bugs4q_returns_bugs4q_parser(factory: DatasetParserFactory) -> None:
    parser = factory.get_parser("bugs4q")

    assert isinstance(parser, Bugs4QDatasetParser)


@pytest.mark.unit
def test_get_parser_thesmellyeight_returns_the_smelly_eight_parser(
    factory: DatasetParserFactory,
) -> None:
    parser = factory.get_parser("thesmellyeight")

    assert isinstance(parser, TheSmellyEightDatasetParser)


@pytest.mark.unit
def test_get_parser_unknown_dataset_raises_value_error(factory: DatasetParserFactory) -> None:
    with pytest.raises(ValueError):
        factory.get_parser("nome_inesistente")


@pytest.mark.unit
def test_get_parser_is_case_insensitive(factory: DatasetParserFactory) -> None:
    parser = factory.get_parser("Bugs4Q")

    assert isinstance(parser, Bugs4QDatasetParser)
