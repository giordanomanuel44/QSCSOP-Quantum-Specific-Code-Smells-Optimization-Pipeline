"""Configurazione di produzione condivisa dei modelli Ollama per gli agenti LLM del MAS.

Punto unico da modificare (nuovo modello, o revisione dell'assemblaggio di produzione): sia
scripts/run_qscsop.py sia i test e2e importano le costanti da qui, nessuna stringa di modello
va duplicata altrove.

DETECTOR_MODEL e' piu' grande di DEFAULT_AGENT_MODEL: vedi
docs/report_detector_agent_model_selection.md per la diagnosi completa che ha motivato la
differenziazione. In sintesi: qwen2.5-coder:14b classifica Long Circuit / Idle Qubits in modo
corretto e consistente dove il 7b confondeva sistematicamente i due smell su un caso di
cancellazione di gate (non ricostruiva la semantica di broadcast di Qiskit), un problema non
risolvibile via prompt engineering sul 7b in quattro tentativi successivi. Il DetectorAgent
richiede un giudizio di classificazione corretto al primo colpo (un falso negativo esclude un
circuito smelly dal ciclo di ottimizzazione), mentre RefactorerAgent e ReviewerAgent tollerano
meglio l'approssimazione, essendo corretti dal ciclo iterativo di validazione/review, e restano
sul modello piu' piccolo e veloce. La dependency injection del modello per-agente, gia' prevista
dall'architettura (CLAUDE.md punto 7), permette questa scelta senza alcuna modifica strutturale.
"""

DETECTOR_MODEL = "ollama/qwen2.5-coder:14b"
DEFAULT_AGENT_MODEL = "ollama/qwen2.5-coder:7b"
