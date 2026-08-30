"""Implementazione concreta del motore matematico di Analytics (sezione 1.6 della tesi)."""

from typing import Optional

import pandas as pd

from qscsop_pipeline.analytics.interfaces.i_metrics_calculator import IMetricsCalculator
from qscsop_pipeline.qscsop.entities.evaluation_status import EvaluationStatus
from qscsop_pipeline.qscsop.mas.detection_thresholds import has_idle_qubits, is_long_circuit
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
            "tasso_successo_globale": self._tasso_successo_globale(processed),
            # Ristretta agli OPTIMIZED, dove prima era la media su tutti i processati. Gli
            # OPT_FAILED non hanno un "numero di iterazioni che e' servito": hanno il budget che
            # hanno esaurito, cioe' maxIterations, sempre lo stesso valore per costruzione.
            # Mediarli insieme ai convergenti impasta una misura con una costante (nella run sul
            # corpus sintetico: 42 OPT_FAILED tutti a 3), e il risultato non dice piu' nulla sulla
            # velocita' di convergenza. Cosi' ristretta risponde a "quando funziona, quanto ci
            # mette", e il tasso di fallimento resta descritto da tasso_risoluzione.
            "iterazioni_alla_convergenza": self._iterazioni_alla_convergenza(optimized_df),
            # Un solo KPI per Long Circuit dove prima ce n'erano due (gateCount e depth fisici).
            # Quelli misuravano il costo del circuito dopo la transpilazione ed erano ciechi ai
            # refactoring della pipeline: su tre casi canonici su tre il delta era zero (vedi
            # docs/misura_metriche_fisiche_pre_rimozione.md), quindi il report avrebbe mostrato
            # ~0% di riduzione su ogni circuito.
            "long_circuit_reduction_pct": self._summary(
                self._pct_reduction(
                    self._get_column(long_circuit_df, "baseline.smellMetrics.longCircuit"),
                    self._get_column(long_circuit_df, "refactored.smellMetrics.longCircuit"),
                ),
                self._get_column(long_circuit_df, "circuitId"),
            ),
            # Riduzione ASSOLUTA e non percentuale: IdQ vale spesso 1 o 2 e il fix riuscito lo
            # porta a 0, dove una percentuale sarebbe sempre 100% e non distinguerebbe un'attesa
            # da sette. Prima questo KPI misurava logicalQubits baseline meno refactored, che
            # sotto QSMELL vale sempre zero: un fix di Idle Qubits non rimuove qubit, li tiene
            # occupati.
            "idle_qubits_reduction": self._summary(
                self._get_column(idle_qubits_df, "baseline.smellMetrics.idleQubits")
                - self._get_column(idle_qubits_df, "refactored.smellMetrics.idleQubits"),
                self._get_column(idle_qubits_df, "circuitId"),
            ),
            # La riduzione non dice se lo smell sia stato RIMOSSO: un circuito puo' scendere da
            # l*c=50 a 26 (-48%) e restare sopra la soglia di 20, quindi ancora smelly. Questo
            # KPI applica ai valori refactored gli stessi predicati con cui il DetectorAgent
            # classifica la baseline, cosi' "risolto" significa la stessa cosa alle due estremita'
            # della pipeline.
            "rientro_sotto_soglia": self._rientro_sotto_soglia(
                optimized_df, long_circuit_df, idle_qubits_df
            ),
            "distribuzione_failure_reason": self._get_column(
                opt_failed_df, "evaluation.failureReason"
            )
            .value_counts()
            .to_dict(),
            "distribuzione_stati": df["evaluation.status"].value_counts().to_dict(),
        }

    @staticmethod
    def _empty_metrics() -> dict:
        """Stessa forma del dict di ritorno di calculate(), tutti i valori None/vuoti.

        Caso limite del dataset di input completamente vuoto (0 record): nessun sottoinsieme
        esiste, quindi ogni metrica e' per definizione non calcolabile.
        """
        empty_summary = {"mean": None, "values": [], "labels": []}
        empty_share = {"count": 0, "total": 0, "rate": None}
        return {
            "tasso_risoluzione": None,
            "tasso_successo_globale": None,
            "iterazioni_alla_convergenza": {"mean": None, "distribution": {}, "n": 0},
            "long_circuit_reduction_pct": dict(empty_summary),
            "idle_qubits_reduction": dict(empty_summary),
            "rientro_sotto_soglia": {
                "long_circuit": dict(empty_share),
                "idle_qubits": dict(empty_share),
                "smell_free": dict(empty_share),
            },
            "distribuzione_failure_reason": {},
            "distribuzione_stati": {},
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

    # RIMOSSA _equivalenza_funzionale. L'equivalenza funzionale non e' un esito misurabile su
    # questo dataset ma un CANCELLO: ValidationService rifiuta di emettere OPTIMIZED se
    # check_equivalence non passa, quindi "quanti OPTIMIZED sono equivalenti" vale 100% per
    # costruzione. La versione precedente aggirava la tautologia allargando il numeratore agli
    # OPT_FAILED con metrics_not_improved, ma quel failureReason descrive solo l'ULTIMO tentativo:
    # un circuito respinto due volte per NOT_EQUIVALENT e uscito al terzo giro su
    # METRICS_NOT_IMPROVED risultava "equivalenza preservata", e viceversa. Con maxIterations=3
    # e' rumore. L'informazione utile -- quanto spesso il refactoring rompe la semantica -- e'
    # gia' nel conteggio di NOT_EQUIVALENT dentro distribuzione_failure_reason.

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

    @classmethod
    def _iterazioni_alla_convergenza(cls, optimized_df: pd.DataFrame) -> dict:
        """Media e distribuzione di iterationCount sui soli circuiti giunti a OPTIMIZED.

        Ritorna anche la DISTRIBUZIONE e non la sola media perche' la variabile e' un intero
        limitato a maxIterations (con maxIterations=3 vive in {1, 2, 3}): una media su cosi'
        pochi valori discreti e' fragile, mentre il conteggio per valore e' leggibile e completo.
        La quota di convergenze al PRIMO tentativo e' il dato che isola il contributo del
        ReviewerAgent -- convergere a 1 significa che non e' mai stato interpellato.
        """
        counts = cls._get_column(optimized_df, "evaluation.iterationCount").dropna()
        if counts.empty:
            return {"mean": None, "distribution": {}, "n": 0}
        return {
            "mean": float(counts.mean()),
            "distribution": {
                int(value): int(occurrences)
                for value, occurrences in counts.value_counts().sort_index().items()
            },
            "n": int(counts.size),
        }

    @classmethod
    def _rientro_sotto_soglia(
        cls,
        optimized_df: pd.DataFrame,
        long_circuit_df: pd.DataFrame,
        idle_qubits_df: pd.DataFrame,
    ) -> dict:
        """Quanti circuiti OPTIMIZED sono scesi sotto la soglia di rilevamento dopo il refactoring.

        Usa i predicati di detection_thresholds e non un confronto scritto qui: le soglie hanno
        un solo punto di definizione nel sistema, e i due operatori NON sono simmetrici
        (Long Circuit inclusivo su l*c >= 20, Idle Qubits stretto su IdQ > 0). Riscriverli a mano
        significherebbe sbagliare l'asimmetria una volta e propagarla in silenzio.

        I denominatori sono diversi per ciascuna voce, deliberatamente: long_circuit e idle_qubits
        sono calcolati sui soli OPTIMIZED in cui QUELLO smell era stato rilevato (non ha senso
        chiedersi se sia rientrato uno smell che non c'era), mentre smell_free guarda tutti gli
        OPTIMIZED e chiede se il circuito sia pulito su ENTRAMBE le metriche.
        """
        refactored_lc = cls._get_column(long_circuit_df, "refactored.smellMetrics.longCircuit")
        refactored_idq = cls._get_column(idle_qubits_df, "refactored.smellMetrics.idleQubits")

        all_lc = cls._get_column(optimized_df, "refactored.smellMetrics.longCircuit")
        all_idq = cls._get_column(optimized_df, "refactored.smellMetrics.idleQubits")
        # Una riga contribuisce a smell_free solo se ENTRAMBE le misure esistono: i predicati
        # sono falsi su NaN (NaN >= 20 e NaN > 0 valgono entrambi False), quindi senza questa
        # maschera un record privo di misura verrebbe contato come pulito.
        measured = all_lc.notna() & all_idq.notna()

        return {
            "long_circuit": cls._share(
                int((~refactored_lc.dropna().apply(is_long_circuit)).sum()),
                int(refactored_lc.dropna().size),
            ),
            "idle_qubits": cls._share(
                int((~refactored_idq.dropna().apply(has_idle_qubits)).sum()),
                int(refactored_idq.dropna().size),
            ),
            "smell_free": cls._share(
                int(
                    (
                        measured & ~all_lc.apply(is_long_circuit) & ~all_idq.apply(has_idle_qubits)
                    ).sum()
                ),
                int(measured.sum()),
            ),
        }

    @staticmethod
    def _share(count: int, total: int) -> dict:
        """{"count", "total", "rate"}; rate None su denominatore nullo, mai una divisione per zero."""
        return {"count": count, "total": total, "rate": (count / total) if total else None}

    @staticmethod
    def _pct_reduction(baseline: pd.Series, new: pd.Series) -> pd.Series:
        """Percentuale di riduzione (baseline-new)/baseline*100.

        NaN dove baseline e' 0: evita una divisione per zero su circuiti degeneri (caso limite,
        ma non escludibile a priori su dataset esterni).
        """
        safe_baseline = baseline.where(baseline != 0)
        return (baseline - new) / safe_baseline * 100

    @staticmethod
    def _summary(values: pd.Series, labels: pd.Series) -> dict:
        """{"mean": float|None, "values": list[float], "labels": list[str]}.

        mean=None su sottoinsieme vuoto (es. nessun OPTIMIZED con un dato smell), mai un crash
        su .mean() di una Series vuota.

        "labels" porta il circuitId di ciascun valore, riallineato per indice DOPO il dropna in
        modo che le due liste restino in corrispondenza posizionale. Serve al ReportVisualizer,
        che disegna una barra per circuito: su sottoinsiemi di poche unita' una barra anonima non
        sarebbe citabile nel testo, mentre con l'etichetta ogni osservazione del grafico e'
        rintracciabile nel dataset.
        """
        clean = values.dropna()
        return {
            "mean": float(clean.mean()) if not clean.empty else None,
            "values": clean.tolist(),
            "labels": labels.reindex(clean.index).fillna("").astype(str).tolist(),
        }

    # RIMOSSA _metriche_per_dataset_source. La valutazione finale gira su un solo corpus (quello
    # sintetico), quindi la ripartizione per datasetSource degenera in un unico gruppo che
    # ripete i valori globali. datasetSource resta nel record come metadato di provenienza.
