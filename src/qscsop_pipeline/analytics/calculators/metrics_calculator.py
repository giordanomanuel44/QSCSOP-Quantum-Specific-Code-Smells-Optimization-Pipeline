"""Implementazione concreta del motore matematico di Analytics (sezione 1.6 della tesi)."""

from typing import Optional

import pandas as pd

from qscsop_pipeline.analytics.interfaces.i_metrics_calculator import IMetricsCalculator
from qscsop_pipeline.qscsop.entities.evaluation_status import EvaluationStatus
from qscsop_pipeline.qscsop.mas.dto.failure_reason import FailureReason
from qscsop_pipeline.qscsop.mas.dto.quantum_smell_type import QuantumSmellType

_PROCESSED_STATUSES = [EvaluationStatus.OPTIMIZED.value, EvaluationStatus.OPT_FAILED.value]


class MetricsCalculator(IMetricsCalculator):
    """Calcola in forma vettorializzata le tre categorie di metriche di sezione 1.6."""

    def calculate(self, df: pd.DataFrame) -> dict:
        """Calcola tutte le metriche di valutazione empirica sul DataFrame ricevuto.

        Un dataset di input vuoto (0 righe) produce da json_normalize([]) un DataFrame privo di
        QUALSIASI colonna, non solo di righe: "evaluation.status" stesso non esiste, quindi ogni
        accesso diretto sotto sollevarebbe KeyError anziché il None/dict-vuoto atteso. Questo
        guard iniziale e' l'unico punto in cui serve controllarlo: superato questo punto,
        "evaluation.status" e' garantito presente (fa parte del contratto di ogni record, mai
        omesso da EvaluationData.to_dict()).
        """
        if "evaluation.status" not in df.columns:
            return self._empty_metrics()

        processed = self._processed_subset(df)

        optimized_df = df[df["evaluation.status"] == EvaluationStatus.OPTIMIZED.value]
        long_circuit_df = optimized_df[
            optimized_df["evaluation.detected_smells"].apply(
                lambda smells: self._has_smell(smells, QuantumSmellType.LONG_CIRCUIT.value)
            )
        ]
        idle_qubits_df = optimized_df[
            optimized_df["evaluation.detected_smells"].apply(
                lambda smells: self._has_smell(smells, QuantumSmellType.IDLE_QUBITS.value)
            )
        ]

        opt_failed_df = df[df["evaluation.status"] == EvaluationStatus.OPT_FAILED.value]

        return {
            "tasso_risoluzione": self._tasso_risoluzione(processed),
            "equivalenza_funzionale": self._equivalenza_funzionale(processed),
            "tasso_successo_globale": self._tasso_successo_globale(processed),
            "numero_medio_iterazioni": self._numero_medio_iterazioni(processed),
            # Un solo KPI per Long Circuit dove prima ce n'erano due (gateCount e depth fisici).
            # Quelli misuravano il costo del circuito dopo la transpilazione ed erano ciechi ai
            # refactoring della pipeline: su tre casi canonici su tre il delta era zero (vedi
            # docs/misura_metriche_fisiche_pre_rimozione.md), quindi il report avrebbe mostrato
            # ~0% di riduzione su ogni circuito.
            "long_circuit_reduction_pct": self._summary(
                self._pct_reduction(
                    self._get_column(long_circuit_df, "baseline.smellMetrics.longCircuit"),
                    self._get_column(long_circuit_df, "refactored.smellMetrics.longCircuit"),
                )
            ),
            # Riduzione ASSOLUTA e non percentuale: IdQ vale spesso 1 o 2 e il fix riuscito lo
            # porta a 0, dove una percentuale sarebbe sempre 100% e non distinguerebbe un'attesa
            # da sette. Prima questo KPI misurava logicalQubits baseline meno refactored, che
            # sotto QSMELL vale sempre zero: un fix di Idle Qubits non rimuove qubit, li tiene
            # occupati.
            "idle_qubits_reduction": self._summary(
                self._get_column(idle_qubits_df, "baseline.smellMetrics.idleQubits")
                - self._get_column(idle_qubits_df, "refactored.smellMetrics.idleQubits")
            ),
            "distribuzione_failure_reason": self._get_column(
                opt_failed_df, "evaluation.failureReason"
            )
            .value_counts()
            .to_dict(),
            "distribuzione_stati": df["evaluation.status"].value_counts().to_dict(),
            "metriche_per_dataset_source": self._metriche_per_dataset_source(df),
        }

    @staticmethod
    def _empty_metrics() -> dict:
        """Stessa forma del dict di ritorno di calculate(), tutti i valori None/vuoti.

        Caso limite del dataset di input completamente vuoto (0 record): nessun sottoinsieme
        esiste, quindi ogni metrica e' per definizione non calcolabile.
        """
        empty_summary = {"mean": None, "values": []}
        return {
            "tasso_risoluzione": None,
            "equivalenza_funzionale": None,
            "tasso_successo_globale": None,
            "numero_medio_iterazioni": None,
            "long_circuit_reduction_pct": dict(empty_summary),
            "idle_qubits_reduction": dict(empty_summary),
            "distribuzione_failure_reason": {},
            "distribuzione_stati": {},
            "metriche_per_dataset_source": {},
        }

    @staticmethod
    def _get_column(df: pd.DataFrame, column: str) -> pd.Series:
        """Ritorna la colonna se presente, altrimenti una Series vuota allineata all'index di df.

        json_normalize crea una colonna solo se almeno un record del dataset la possiede: un
        sottoinsieme senza alcun OPTIMIZED (o alcun fallimento) non ha semplicemente NaN in
        refactored.*/failureReason, ne e' del tutto privo -- un accesso diretto solleverebbe
        KeyError, non un NaN silenzioso.
        """
        return df[column] if column in df.columns else pd.Series(dtype="float64", index=df.index)

    @staticmethod
    def _has_smell(detected_smells, smell: str) -> bool:
        return isinstance(detected_smells, list) and smell in detected_smells

    @staticmethod
    def _processed_subset(df: pd.DataFrame) -> pd.DataFrame:
        """Circuiti che hanno attraversato il ciclo di ottimizzazione (esclude SMELL_FREE).

        SMELL_FREE ha iterationCount=0 per costruzione: includerlo falserebbe sia i tassi di
        Categoria 1/3 sia la media iterazioni di Categoria 3 -- stesso motivo per entrambe,
        sottoinsieme condiviso per non duplicare la stessa logica di filtro due volte.
        """
        return df[df["evaluation.status"].isin(_PROCESSED_STATUSES)]

    @classmethod
    def _tasso_risoluzione(cls, processed: pd.DataFrame) -> Optional[float]:
        if processed.empty:
            return None
        return float((processed["evaluation.status"] == EvaluationStatus.OPTIMIZED.value).mean())

    @classmethod
    def _equivalenza_funzionale(cls, processed: pd.DataFrame) -> Optional[float]:
        """OPTIMIZED oppure OPT_FAILED con failureReason=metrics_not_improved: il circuito ha
        comunque superato il controllo di equivalenza, fallendo solo su "Migliori?". Esclude
        NOT_EQUIVALENT, COMPILATION_FAILED e, per scelta conservativa (stesso spirito del
        criterio Pareto in ValidationService), anche UNEXPECTED_ERROR: un crash imprevisto non
        garantisce che l'equivalenza sia mai stata verificata.
        """
        if processed.empty:
            return None
        optimized = processed["evaluation.status"] == EvaluationStatus.OPTIMIZED.value
        preserved = optimized | (
            cls._get_column(processed, "evaluation.failureReason")
            == FailureReason.METRICS_NOT_IMPROVED.value
        )
        return float(preserved.mean())

    @classmethod
    def _tasso_successo_globale(cls, processed: pd.DataFrame) -> Optional[float]:
        """Metrica piu' permissiva della terza categoria: include anche NOT_EQUIVALENT (codice
        valido e compilabile, solo semanticamente diverso). Esclude SOLO COMPILATION_FAILED e,
        per la stessa scelta conservativa di _equivalenza_funzionale, UNEXPECTED_ERROR.
        """
        if processed.empty:
            return None
        optimized = processed["evaluation.status"] == EvaluationStatus.OPTIMIZED.value
        valid_but_failed = cls._get_column(processed, "evaluation.failureReason").isin(
            [FailureReason.NOT_EQUIVALENT.value, FailureReason.METRICS_NOT_IMPROVED.value]
        )
        return float((optimized | valid_but_failed).mean())

    @staticmethod
    def _numero_medio_iterazioni(processed: pd.DataFrame) -> Optional[float]:
        if processed.empty:
            return None
        return float(processed["evaluation.iterationCount"].mean())

    @staticmethod
    def _pct_reduction(baseline: pd.Series, new: pd.Series) -> pd.Series:
        """Percentuale di riduzione (baseline-new)/baseline*100.

        NaN dove baseline e' 0: evita una divisione per zero su circuiti degeneri (caso limite,
        ma non escludibile a priori su dataset esterni).
        """
        safe_baseline = baseline.where(baseline != 0)
        return (baseline - new) / safe_baseline * 100

    @staticmethod
    def _summary(values: pd.Series) -> dict:
        """{"mean": float|None, "values": list[float]}.

        mean=None su sottoinsieme vuoto (es. nessun OPTIMIZED con un dato smell nei 13 circuiti
        reali), mai un crash su .mean() di una Series vuota. "values" alimenta il boxplot/
        histogram del ReportVisualizer.
        """
        clean = values.dropna()
        return {
            "mean": float(clean.mean()) if not clean.empty else None,
            "values": clean.tolist(),
        }

    @classmethod
    def _metriche_per_dataset_source(cls, df: pd.DataFrame) -> dict:
        """Categoria 1 calcolata separatamente per ciascun datasetSource presente nel df."""
        return {
            source: {
                "tasso_risoluzione": cls._tasso_risoluzione(
                    cls._processed_subset(df[df["datasetSource"] == source])
                ),
                "equivalenza_funzionale": cls._equivalenza_funzionale(
                    cls._processed_subset(df[df["datasetSource"] == source])
                ),
            }
            for source in df["datasetSource"].dropna().unique()
        }
