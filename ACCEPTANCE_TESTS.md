# Critères d’acceptation

## A. Installation

### A1

```bash
uv sync
```

doit réussir sur Python 3.12.

### A2

```bash
uv run automl --help
```

doit afficher les commandes principales.

### A3

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

doivent réussir.

## B. Classification

### B1 — Classification binaire

La commande :

```bash
uv run automl fit examples/binary_train.csv \
  --target target \
  --task binary_classification \
  --budget 60s \
  --output runs/binary
```

doit :

- terminer proprement ;
- exécuter une baseline ;
- exécuter plusieurs familles ;
- produire un leaderboard ;
- sauvegarder un pipeline ;
- produire des prédictions OOF ;
- produire un notebook.

### B2 — Multiclasse

Même comportement sur un dataset multiclasse synthétique.

### B3 — Catégories inconnues

Une catégorie présente uniquement dans un fold de validation ou dans le fichier test ne doit pas faire échouer l’inférence.

## C. Régression

### C1

La commande :

```bash
uv run automl fit examples/regression_train.csv \
  --target target \
  --task regression \
  --budget 60s \
  --output runs/regression
```

doit produire un pipeline dont la RMSE est meilleure que celle du DummyRegressor sur le dataset contrôlé.

## D. Valeurs manquantes

### D1

Les datasets avec valeurs manquantes numériques et catégorielles doivent être supportés.

### D2

Au moins trois stratégies numériques et deux stratégies catégorielles doivent être réellement évaluées lorsque compatibles.

### D3

L’imputer doit être ajusté à l’intérieur du fold.

Test attendu : une statistique artificiellement différente entre train et validation ne doit pas être visible lors du fit du fold.

## E. Détection de fuite

### E1

Une colonne exactement égale à la cible doit être signalée comme fuite critique.

### E2

Une colonne identifiant presque unique doit être signalée.

### E3

Des doublons exacts traversant des folds doivent être signalés.

### E4

Lorsque `group_column` est fourni, aucun groupe ne doit être réparti entre entraînement et validation.

### E5

Le pipeline final ne doit jamais inclure la cible dans `X`.

## F. Reproductibilité

### F1

Deux runs avec :

- mêmes données ;
- même configuration ;
- même seed ;
- même environnement ;

doivent produire :

- les mêmes folds ;
- les mêmes candidats initiaux ;
- le même meilleur `PipelineSpec` lorsque les modèles sont déterministes ;
- des scores dans une tolérance définie sinon.

### F2

Le manifeste doit enregistrer les versions de Python et des dépendances principales.

## G. Résilience

### G1

Un pipeline volontairement invalide doit être marqué `failed` sans arrêter tout le run.

### G2

Un timeout d’essai doit être enregistré.

### G3

Une dépendance optionnelle absente doit désactiver uniquement les modèles correspondants.

### G4

Un dataset invalide doit produire une erreur claire et un code de sortie non nul.

## H. Reprise

### H1

Interrompre un run après plusieurs essais, puis exécuter :

```bash
uv run automl resume runs/binary
```

doit reprendre le run.

### H2

Les essais terminés ne doivent pas être répétés.

### H3

Un hash de dataset différent doit empêcher une reprise silencieuse.

### H4

Le scheduler et l’allocateur doivent restaurer leur état.

## I. Multi-fidélité

### I1

Le journal doit montrer des essais aux différents niveaux de fidélité.

### I2

Seule une partie des essais faibles doit être promue.

### I3

Les finalistes doivent être réévalués à fidélité complète.

### I4

Le système doit réserver du temps pour l’entraînement final.

## J. Allocation adaptative

### J1

Chaque famille compatible doit recevoir une exploration minimale.

### J2

Sur un scénario synthétique où une famille produit de meilleures récompenses à moindre coût, elle doit recevoir davantage d’essais.

### J3

Une famille qui échoue répétitivement doit être temporairement suspendue.

### J4

La politique doit être testée sans dépendre d’un vrai entraînement coûteux.

## K. Artefacts

Le dossier de run doit contenir :

```text
manifest.json
dataset_profile.json
validation_plan.json
leaderboard.csv
trials.parquet ou trials.csv
best_pipeline_spec.json
best_pipeline.joblib
solution.ipynb
report.html
environment.json
logs/
```

Lorsque test CSV fourni :

```text
predictions.csv
```

Lorsque OOF disponible :

```text
oof_predictions.parquet ou oof_predictions.csv
```

## L. Notebook

### L1

Le notebook doit s’exécuter automatiquement avec `nbclient`.

### L2

Il ne doit nécessiter aucune saisie manuelle.

### L3

Il doit reconstruire ou charger le pipeline réellement sélectionné.

### L4

Ses prédictions doivent correspondre à celles du pipeline sauvegardé :

- égalité exacte pour sorties déterministes ;
- tolérance numérique documentée sinon.

### L5

Le manifeste doit enregistrer :

```json
{
  "notebook_validation": {
    "status": "passed",
    "executed_at": "...",
    "prediction_match": true
  }
}
```

## M. API Python

Ce code doit fonctionner :

```python
from autonomous_automl import AutoMLConfig, AutoMLRun

config = AutoMLConfig(
    target="target",
    task="auto",
    metric="auto",
    budget_seconds=60,
    random_seed=42,
    output_dir="runs/api-test",
)

result = AutoMLRun(config).fit("examples/binary_train.csv")

assert result.best_pipeline_path.exists()
assert result.manifest_path.exists()
```

## N. Performance minimale

Sur les datasets synthétiques contrôlés :

- le meilleur pipeline doit battre la baseline naïve ;
- la recherche ne doit pas dépasser le budget de plus de 20 %, hors installation ;
- les tests standards doivent rester raisonnablement rapides ;
- aucun benchmark lourd ne doit s’exécuter par défaut.

## O. Definition of Done finale

La V1 est acceptée uniquement si :

- A à N sont couverts par des tests automatisés ou des validations explicites ;
- tous les tests passent ;
- aucun stub critique ne subsiste ;
- les commandes documentées fonctionnent ;
- le notebook est validé ;
- la reprise fonctionne ;
- le scheduler adaptatif est réellement connecté au moteur ;
- la mémoire est réellement consultée pour un warm start lorsqu’un historique existe.
