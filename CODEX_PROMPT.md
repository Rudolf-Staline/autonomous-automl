# Prompt maître à donner à Codex

Tu es responsable de l’implémentation complète de ce projet dans le dépôt courant.

Le dépôt contient pour le moment des documents de spécification. Tu dois transformer ces documents en une application Python fonctionnelle, testée, reproductible et documentée.

## Ordre de lecture obligatoire

Lis intégralement, dans cet ordre :

1. `AGENTS.md`
2. `SPEC.md`
3. `ARCHITECTURE.md`
4. `TECHNICAL_DECISIONS.md`
5. `IMPLEMENTATION_PLAN.md`
6. `ACCEPTANCE_TESTS.md`
7. `BENCHMARK_PLAN.md`
8. `IMPLEMENTATION_STATUS.md`

Ne commence pas l’implémentation avant d’avoir compris les contrats, les invariants et la Definition of Done.

## Mission

Construis intégralement un moteur AutoML autonome pour données tabulaires.

Le système reçoit un ou plusieurs fichiers CSV, la colonne cible, le type de tâche ou `auto`, une métrique ou `auto`, un budget de calcul et des contraintes matérielles.

Il doit :

- charger et valider les données ;
- profiler le dataset ;
- détecter les colonnes suspectes, identifiants et risques de fuite ;
- choisir une stratégie de validation ;
- générer uniquement des pipelines compatibles ;
- tester plusieurs stratégies d’imputation, d’encodage et de modélisation ;
- optimiser intelligemment les hyperparamètres ;
- utiliser une recherche multi-fidélité ;
- allouer adaptativement le budget entre familles de pipelines ;
- journaliser tous les essais ;
- survivre aux erreurs de pipelines individuels ;
- reprendre un run interrompu ;
- sélectionner et réentraîner le pipeline final ;
- sauvegarder tous les artefacts ;
- générer un notebook déterministe à partir du manifeste réel ;
- exécuter ce notebook dans un environnement propre ;
- vérifier que les prédictions reproduites correspondent aux artefacts du run.

## Règles de travail

1. Inspecte le dépôt avant toute modification.
2. Mets à jour `IMPLEMENTATION_STATUS.md` avec :
   - le plan courant ;
   - le milestone actif ;
   - les tâches terminées ;
   - les décisions prises ;
   - les tests exécutés ;
   - les échecs connus.
3. Implémente les milestones dans l’ordre de dépendance défini dans `IMPLEMENTATION_PLAN.md`.
4. Après chaque milestone :
   - lance les tests correspondants ;
   - corrige les régressions ;
   - mets à jour le statut.
5. Ne crée pas de faux composants décoratifs.
6. Aucun composant critique ne doit rester sous forme de stub, pseudo-code, mock permanent ou fonction non connectée.
7. Aucun `TODO` critique ne doit subsister dans le chemin principal.
8. Le code doit être typé et lisible.
9. Les erreurs d’un modèle ou d’un pipeline doivent être journalisées sans arrêter tout le run.
10. Tous les résultats doivent être reproductibles avec une seed fixe.
11. Le notebook final doit être généré à partir du manifeste et des objets réellement utilisés.
12. Toute transformation apprise doit être ajustée uniquement sur les folds d’entraînement.
13. Le test final et le fichier de prédiction ne doivent jamais influencer la recherche.
14. Ne t’arrête pas après la création du squelette.
15. Continue jusqu’à satisfaction de la Definition of Done.

## Libertés autorisées

Tu peux :

- créer et modifier tous les fichiers nécessaires ;
- initialiser Git si utile ;
- créer un environnement Python ;
- installer les dépendances ;
- exécuter des commandes ;
- écrire des scripts de migration ;
- générer de petits datasets synthétiques ;
- refactoriser une architecture qui bloque l’avancement ;
- consulter les documentations officielles des bibliothèques ;
- remplacer une bibliothèque si elle est techniquement incompatible, à condition de documenter la décision ;
- réduire temporairement les budgets des tests pour garder la suite rapide.

## Contraintes techniques minimales

- Python 3.12 ;
- gestion du projet avec `uv` ;
- structure `src/` ;
- `pytest` ;
- `ruff` ;
- `pyright` ;
- scikit-learn ;
- Optuna ;
- pandas ou Polars, avec une justification documentée ;
- SQLite pour le registre initial ;
- Joblib pour la sérialisation du pipeline ;
- `nbformat` et `nbclient` pour le notebook ;
- CLI basée sur Typer ;
- configuration sérialisable en JSON ;
- logs structurés ;
- aucune interface web dans la V1.

## Invariants non négociables

- Pas de fuite entre entraînement et validation.
- L’imputation et l’encodage sont inclus dans les pipelines de validation.
- Le target encoding, s’il est implémenté, est cross-fitté.
- Les catégories inconnues ne font pas échouer l’inférence.
- Les pipelines invalides sont filtrés avant exécution lorsque possible.
- Les essais échoués sont conservés dans le registre.
- Le meilleur pipeline est reconstruisible depuis son `PipelineSpec`.
- Un run interrompu peut reprendre sans perdre les essais terminés.
- Le notebook doit reproduire le pipeline final, pas seulement montrer un exemple similaire.
- Les prédictions du notebook doivent correspondre aux prédictions enregistrées dans la tolérance définie.
- Le système doit toujours inclure une baseline naïve.
- Les contraintes de budget doivent être respectées avec une marge raisonnable.
- Les modèles optionnels absents ne doivent pas casser le noyau du produit.

## Attendus d’implémentation

Le dépôt final doit contenir notamment :

```text
pyproject.toml
src/autonomous_automl/
tests/
examples/
docs/
scripts/
README.md
CHANGELOG.md
LICENSE
```

Le package doit exposer au minimum :

```python
from autonomous_automl import AutoMLRun, AutoMLConfig

run = AutoMLRun(
    AutoMLConfig(
        target="target",
        task="auto",
        metric="auto",
        budget_seconds=1800,
        random_seed=42,
    )
)

result = run.fit("data/train.csv")
```

Le CLI doit exposer au minimum :

```bash
automl fit
automl resume
automl inspect
automl leaderboard
automl export-notebook
automl validate-artifacts
```

## Definition of Done

Le travail n’est terminé que si :

- tous les tests unitaires passent ;
- tous les tests d’intégration passent ;
- Ruff passe ;
- Pyright passe sur le code de production ;
- le CLI fonctionne sur les datasets d’exemple ;
- un run interrompu peut reprendre ;
- le meilleur pipeline est sérialisé ;
- le notebook généré s’exécute depuis zéro ;
- les prédictions reproduites correspondent aux artefacts ;
- les risques de fuite synthétiques sont détectés ou correctement neutralisés ;
- la documentation permet à un développeur externe d’installer et d’utiliser le projet ;
- aucun composant critique n’est simulé ;
- `IMPLEMENTATION_STATUS.md` indique clairement que tous les critères sont satisfaits.

## Première action

Commence maintenant par :

1. inspecter tous les documents ;
2. reformuler l’architecture dans `IMPLEMENTATION_STATUS.md` ;
3. produire un plan de fichiers et de milestones ;
4. identifier les risques techniques ;
5. commencer le milestone M0.

Ne demande pas une confirmation après le plan. Poursuis l’implémentation.
