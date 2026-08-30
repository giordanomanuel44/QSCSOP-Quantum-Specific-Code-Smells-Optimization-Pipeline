"""Configurazione di produzione del modello Ollama condiviso dagli agenti LLM del MAS.

Punto unico da modificare (nuovo modello, o revisione dell'assemblaggio di produzione): sia
scripts/run_qscsop.py sia i test e2e importano la costante da qui, nessuna stringa di modello
va duplicata altrove.

UN SOLO MODELLO PER TUTTI E TRE GLI AGENTI. Questa e' una revisione della strategia
differenziata precedente (DETECTOR_MODEL su qwen2.5-coder:14b, DEFAULT_AGENT_MODEL su
qwen2.5-coder:7b), documentata in docs/report_detector_agent_model_selection.md: quel report
resta valido come diagnosi storica -- il 7b confondeva sistematicamente i due smell su un caso
di cancellazione di gate -- ma la sua premessa non regge piu'. La diagnosi riguardava un
DetectorAgent che CLASSIFICAVA; da allora la classificazione e' uscita dall'LLM ed e' passata a
QiskitFacade.calculate_smell_metrics piu' detection_thresholds, che sono esatte per costruzione.
All'LLM resta la sola prescrizione, e il divario di capacita' che motivava due modelli diversi
non si applica a quel compito.

Con qwen3-coder:30b, inoltre, il modello piu' piccolo della coppia originale non e' piu' in
esercizio: mantenere due costanti con lo stesso valore documentava una differenziazione che
nei fatti non esisteva.

L'architettura continua a permettere modelli per-agente senza modifiche strutturali: gli agenti
dipendono da crewai.BaseLLM e ricevono l'istanza via costruttore, quindi reintrodurre una
seconda costante e' una modifica locale a questo file e ai punti di assemblaggio.
"""

DEFAULT_AGENT_MODEL = "ollama/qwen3-coder:30b"
