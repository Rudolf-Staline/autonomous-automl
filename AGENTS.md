# Instructions permanentes pour les agents

Ce fichier contient les règles de travail permanentes du dépôt.

## Priorités

Par ordre décroissant :

1. exactitude de la validation ;
2. absence de fuite ;
3. reproductibilité ;
4. robustesse aux erreurs ;
5. simplicité du noyau ;
6. performance ;
7. extensibilité ;
8. ergonomie.

Une amélioration de score ne justifie jamais une fuite de données, une perte de reproductibilité ou une architecture impossible à tester.

## Style de développement

- Préférer des composants petits et testables.
- Séparer les descriptions sérialisables des objets exécutables.
- Ne jamais stocker directement un estimateur comme source de vérité.
- La source de vérité d’un pipeline est son `PipelineSpec`.
- Toute décision automatique importante doit être explicable et enregistrée.
- Les dépendances optionnelles doivent être isolées derrière des adapters.
- Les fonctions publiques doivent être typées.
- Les erreurs doivent avoir des messages exploitables.
- Les logs ne doivent jamais contenir les données brutes de l’utilisateur.
- Les secrets et chemins personnels ne doivent jamais être commités.

## Arborescence attendue

```text
src/autonomous_automl/
├── api/
├── cli/
├── contracts/
├── data/
├── profiling/
├── validation/
├── components/
├── pipelines/
├── search/
├── evaluation/
├── tracking/
├── memory/
├── reporting/
├── runtime/
└── utils/
```

## Contrats

Les contrats centraux doivent être définis dans `contracts/` :

- `AutoMLConfig`
- `DatasetBundle`
- `DatasetProfile`
- `ValidationPlan`
- `PipelineSpec`
- `FidelitySpec`
- `TrialResult`
- `RunManifest`
- `RunResult`

Ces objets doivent être :

- typés ;
- validés ;
- sérialisables en JSON ;
- versionnés lorsque leur format est persistant.

## Tests

Les tests sont répartis en :

```text
tests/
├── unit/
├── integration/
├── leakage/
├── reproducibility/
├── resume/
├── notebook/
└── benchmarks/
```

Toute correction d’un bug important doit ajouter un test de non-régression.

## Commandes de qualité

Le projet doit fournir des commandes simples :

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

Une commande agrégée peut être ajoutée :

```bash
uv run python scripts/quality.py
```

## Données de test

- Utiliser de petits datasets synthétiques pour la majorité des tests.
- Ne pas télécharger de gros datasets pendant la suite standard.
- Les benchmarks lourds doivent être explicitement séparés.
- Les datasets synthétiques doivent couvrir :
  - données numériques ;
  - données catégorielles ;
  - valeurs manquantes ;
  - catégories inconnues ;
  - forte cardinalité ;
  - classes déséquilibrées ;
  - doublons ;
  - fuite de cible ;
  - group leakage ;
  - colonne temporelle.

## Sécurité

- Ne jamais exécuter du code fourni dans les CSV.
- Ne jamais utiliser `eval` sur des entrées utilisateur.
- Limiter la lecture aux fichiers explicitement fournis.
- Éviter les sérialisations non sûres pour les données externes.
- Joblib est autorisé uniquement pour les artefacts produits localement par le système.
- Les rapports et notebooks doivent échapper les chaînes issues des noms de colonnes.

## Gestion des dépendances

Le noyau doit fonctionner avec les dépendances obligatoires.

Les modèles lourds peuvent être optionnels :

- XGBoost ;
- LightGBM ;
- CatBoost.

Le système doit :

- détecter leur disponibilité ;
- enregistrer les modèles indisponibles ;
- continuer avec le registre minimal ;
- ne jamais échouer au démarrage à cause d’une dépendance optionnelle.

## Politique de changement

Avant une modification architecturale importante :

1. expliquer le problème ;
2. documenter la décision ;
3. préserver la compatibilité des manifestes si possible ;
4. ajouter ou modifier les tests ;
5. mettre à jour `IMPLEMENTATION_STATUS.md`.
