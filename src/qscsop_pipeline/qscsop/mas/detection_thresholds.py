"""Soglie di rilevamento dei due Quantum Code Smell in scope, con i rispettivi predicati.

Punto unico da modificare: nessun numero di soglia va duplicato altrove. Prima di questo modulo
il taglio di Long Circuit era scritto tre volte (prompts.py, corpus_reliability_report.py,
qsmell_metrics_report.py) e quello di Idle Qubits compariva inline in quattro punti.

PERCHE' QUI E NON NELLA FACADE. La facade MISURA (calculate_smell_metrics ritorna l, c, l*c e
IdQ); il confronto con una soglia e' politica di rilevamento e appartiene al livello che decide,
cioe' il MAS. E' la stessa separazione gia' in atto fra QiskitFacade.calculate_metrics e
ValidationService._is_improvement.

PERCHE' DEI PREDICATI E NON SOLO LE COSTANTI. I due confronti non hanno lo stesso operatore:
Long Circuit e' INCLUSIVO (l*c >= 20, un circuito esattamente a 20 e' gia' smelly), Idle Qubits
e' STRETTO (IdQ > 0, una sola cella di attesa basta). Esportare solo i numeri lascerebbe a ogni
chiamante il compito di ricordare l'asimmetria, ed e' il tipo di dettaglio che si sbaglia una
volta e poi si propaga in silenzio nelle etichette.
"""

# Taglio di Long Circuit: la soglia pubblicata da Chen et al. (ICSE 2023) e' 0.50 sulla forma
# esponenziale (1 - error)^(l*c), calcolata con l'errore di gate della IBM Kolkata di agosto
# 2022. Quel criterio equivale esattamente a l*c >= 20, che ne e' la forma machine-independent:
# non dipende dal device e resta confrontabile fra circuiti misurati su backend diversi.
LC_PRODUCT_CUTOFF = 20

# Taglio di Idle Qubits: il valore pubblicato e' 0.00, ottenuto dagli autori come mediana della
# metrica sul loro corpus. Un qubit che attende anche una sola colonna fra due sue operazioni
# consecutive e' quindi gia' smelly.
IDLE_QUBITS_CUTOFF = 0


def is_long_circuit(lc_product: int) -> bool:
    """True se il prodotto l*c misurato raggiunge il taglio di Long Circuit (confronto inclusivo)."""
    return lc_product >= LC_PRODUCT_CUTOFF


def has_idle_qubits(idq: int) -> bool:
    """True se la metrica IdQ misurata supera il taglio di Idle Qubits (confronto stretto)."""
    return idq > IDLE_QUBITS_CUTOFF
