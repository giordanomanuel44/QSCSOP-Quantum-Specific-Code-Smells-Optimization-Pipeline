"""Entità di dominio pura: la misura QSMELL dei due smell in scope su una versione di circuito."""


class SmellMetrics:
    """Le tre grandezze che QSMELL misura su un circuito: l, c e IdQ.

    Sostituisce CircuitMetrics, che portava gateCount e depth in due istanze separate (astratte e
    fisiche). Quelle misuravano il COSTO del circuito, non gli smell, ed erano cieche ai
    refactoring in esame -- vedi docs/misura_metriche_fisiche_pre_rimozione.md per i numeri.

    PERCHE' l E c SEPARATI E NON IL SOLO PRODOTTO. Due circuiti con lo stesso l*c possono avere
    forma opposta: nel dataset sintetico wide_layers_1 e' l=10 c=4 e large_mixed_3 e' l=8 c=5,
    entrambi 40, ma uno e' alto e stretto e l'altro largo e basso -- si riparano in modi diversi.
    E senza c non e' piu' calcolabile quanto un circuito debba rimpicciolirsi per tornare sotto
    soglia (l_max = (soglia - 1) // c), che e' la stratificazione con cui si distingue un
    fallimento del sistema da un circuito non riparabile per costruzione.

    PERCHE' long_circuit E' UNA PROPERTY E NON UN CAMPO. E' esattamente l * c, quindi memorizzarlo
    creerebbe uno stato in cui long_circuit != l * c: un'incoerenza che nessun controllo puo'
    impedire e che prima o poi si verifica. Non si memorizza cio' che si puo' derivare.
    """

    def __init__(self, max_ops_per_qubit: int, max_parallel_ops: int, idle_qubits: int) -> None:
        self._max_ops_per_qubit = max_ops_per_qubit
        self._max_parallel_ops = max_parallel_ops
        self._idle_qubits = idle_qubits

    def get_max_ops_per_qubit(self) -> int:
        """Il numero di operazioni sul qubit piu' carico ("l" nella notazione del paper)."""
        return self._max_ops_per_qubit

    def set_max_ops_per_qubit(self, value: int) -> None:
        self._max_ops_per_qubit = value

    def get_max_parallel_ops(self) -> int:
        """Il numero di operazioni nella colonna piu' affollata ("c" nella notazione del paper)."""
        return self._max_parallel_ops

    def set_max_parallel_ops(self, value: int) -> None:
        self._max_parallel_ops = value

    def get_idle_qubits(self) -> int:
        """La piu' lunga attesa di un qubit fra due sue operazioni consecutive (IdQ)."""
        return self._idle_qubits

    def set_idle_qubits(self, value: int) -> None:
        self._idle_qubits = value

    @property
    def long_circuit(self) -> int:
        """La metrica Long Circuit, l * c. Derivata ad ogni accesso: non esiste un setter."""
        return self._max_ops_per_qubit * self._max_parallel_ops

    def to_dict(self) -> dict:
        """Serializza le tre grandezze piu' la metrica derivata.

        longCircuit COMPARE nel dict pur essendo derivato: qui non si sta esponendo stato
        mutabile ma una proiezione di sola lettura per i consumatori a valle (Analytics lo vuole
        come colonna del DataFrame, senza doverlo ricalcolare). L'invariante resta protetto perche'
        nessun setter lo attraversa.
        """
        return {
            "maxOpsPerQubit": self._max_ops_per_qubit,
            "maxParallelOps": self._max_parallel_ops,
            "longCircuit": self.long_circuit,
            "idleQubits": self._idle_qubits,
        }
