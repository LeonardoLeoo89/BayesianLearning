# Bayesian Learning

Una piattaforma unificata per l'apprendimento di reti bayesiane e SEM.

## Installazione

Le dipendenze del progetto sono gestite con uv.

```bash
# 1. Clone
git clone https://github.com/LeonardoLeoo89/BayesianLearning.git
cd BayesianLearning

# 2. Setup ambiente e dipendenze
uv sync

# 3. Informazioni sullo usage 
uv run bayesian-learn --help
```

## Utilizzo con Docker

È possibile eseguire l'intero progetto tramite Docker, evitando l'installazione delle dipendenze di sistema (come Java) sul computer host.

```bash
# 1. Costruzione dell'immagine Docker
docker build -t bayesian-learning .

# 2. Esecuzione del tool da riga di comando
docker run --rm bayesian-learning --help

# 3. (Opzionale) Esecuzione degli script con montaggio dei risultati in locale
# È possibile salvare i grafici e i dati in locale montando un volume durante l'esecuzione:
docker run --rm -v $(pwd)/results:/app/results bayesian-learning uv run python scripts/benchmarks/evaluate_accuracy.py
```

## Documentazione

Una versione HTML della documentazione è consultabile in `docs/_build`, generata
a partire dalle docstring che corredano il codice.
