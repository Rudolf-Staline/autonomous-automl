# Autonomous AutoML Engine

Ce dépôt contient la spécification et les règles de construction d’un moteur AutoML autonome pour données tabulaires.

Le système doit recevoir un ou plusieurs fichiers CSV, comprendre automatiquement le problème, construire un espace de recherche de pipelines compatibles, allouer intelligemment le budget de calcul, sélectionner le meilleur pipeline et générer un notebook entièrement reproductible.

## Démarrage avec Codex

1. Place tous les fichiers de ce pack à la racine du dossier vide.
2. Ouvre ce dossier dans Codex.
3. Donne à Codex le contenu de `CODEX_PROMPT.md`.
4. Laisse Codex inspecter les documents avant qu’il ne commence à coder.
5. Ne lui demande pas seulement de créer le squelette : la Definition of Done est définie dans `ACCEPTANCE_TESTS.md`.

Ordre de lecture imposé à Codex :

1. `AGENTS.md`
2. `SPEC.md`
3. `ARCHITECTURE.md`
4. `TECHNICAL_DECISIONS.md`
5. `IMPLEMENTATION_PLAN.md`
6. `ACCEPTANCE_TESTS.md`
7. `BENCHMARK_PLAN.md`
8. `IMPLEMENTATION_STATUS.md`

## Objectif de la première version

La V1 doit prendre en charge :

- classification binaire ;
- classification multiclasse ;
- régression ;
- fichiers CSV tabulaires ;
- colonnes numériques, catégorielles, booléennes et dates simples ;
- valeurs manquantes ;
- validation croisée sans fuite ;
- recherche conditionnelle de pipelines ;
- tuning Optuna ;
- recherche multi-fidélité ;
- allocation adaptative entre familles de modèles ;
- sauvegarde et reprise des expériences ;
- génération et exécution automatique d’un notebook final.

La V1 ne prend pas en charge :

- séries temporelles complètes ;
- NLP ;
- vision ;
- apprentissage distribué multi-nœuds ;
- interface web ;
- déploiement cloud ;
- modèles de fondation.

## Commande cible

```bash
automl fit data/train.csv \
  --target target \
  --task auto \
  --metric auto \
  --budget 30m \
  --output runs/demo
```

Sorties attendues :

```text
runs/demo/
├── manifest.json
├── dataset_profile.json
├── leaderboard.csv
├── trials.parquet
├── best_pipeline.joblib
├── predictions.csv
├── solution.ipynb
├── report.html
├── requirements-lock.txt
└── logs/
```

## Principe fondamental

Le produit n’est pas un simple wrapper autour d’Optuna.

La différenciation repose sur :

- le profilage structuré du dataset ;
- les règles de compatibilité entre données, imputations, encodages et modèles ;
- l’allocation adaptative du budget ;
- la promotion multi-fidélité ;
- la mémoire des expériences ;
- la détection des risques de fuite ;
- la reproductibilité certifiée du notebook généré.
