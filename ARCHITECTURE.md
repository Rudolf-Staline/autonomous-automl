# Architecture technique

## 1. Vue générale

```text
CSV
 │
 ▼
DataLoader
 │
 ▼
DatasetProfiler ───────► LeakageDetector
 │                         │
 ▼                         ▼
DatasetProfile       LeakageReport
 │
 ▼
ValidationPlanner
 │
 ▼
PipelinePlanner
 │
 ▼
PipelineGrammar + ModelRegistry
 │
 ▼
AdaptiveSearchController
 │
 ├── FamilyAllocator
 ├── FidelityScheduler
 ├── OptunaFamilyOptimizer
 └── BudgetManager
 │
 ▼
PipelineEvaluator
 │
 ├── cross-validation
 ├── metrics
 ├── resource monitoring
 └── failure isolation
 │
 ▼
ExperimentStore
 │
 ├── SQLite
 ├── artifacts
 └── resume state
 │
 ▼
FinalSelector
 │
 ▼
FinalTrainer
 │
 ▼
NotebookCompiler
 │
 ▼
ArtifactValidator
```

## 2. Principe de séparation

Trois niveaux doivent rester séparés :

### Description

Objets sérialisables :

- configuration ;
- profil ;
- plan de validation ;
- pipeline ;
- fidélité ;
- résultat ;
- manifeste.

### Construction

Factories déterministes :

- `build_preprocessor(spec, profile)` ;
- `build_model(spec, profile)` ;
- `build_pipeline(spec, profile)`.

### Exécution

Services :

- profilage ;
- validation ;
- évaluation ;
- optimisation ;
- stockage ;
- génération.

## 3. Contrats principaux

### AutoMLConfig

```python
class AutoMLConfig:
    target: str
    task: str
    metric: str
    budget_seconds: int
    random_seed: int
    n_jobs: int
    output_dir: str
    test_path: str | None
    group_column: str | None
    time_column: str | None
    id_column: str | None
    optimization_profile: str
    enable_gpu: bool
    memory_limit_mb: int | None
```

### DatasetBundle

```python
class DatasetBundle:
    X: DataFrame
    y: Series
    test_X: DataFrame | None
    row_ids: Series | None
    source_hashes: dict[str, str]
```

### DatasetProfile

```python
class DatasetProfile:
    n_rows: int
    n_features: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    boolean_columns: list[str]
    datetime_columns: list[str]
    text_columns: list[str]
    constant_columns: list[str]
    id_candidates: list[str]
    missing_ratios: dict[str, float]
    cardinalities: dict[str, int]
    unique_ratios: dict[str, float]
    target_profile: dict[str, object]
    landmarks: dict[str, float]
    estimated_memory_mb: float
```

### ValidationPlan

```python
class ValidationPlan:
    splitter_name: str
    n_splits: int
    shuffle: bool
    random_seed: int | None
    group_column: str | None
    time_column: str | None
    predefined_fold_column: str | None
    rationale: list[str]
```

### PipelineSpec

```python
class PipelineSpec:
    spec_version: int
    family: str
    numeric_imputer: str
    numeric_scaler: str
    categorical_imputer: str
    categorical_encoder: str
    datetime_transformer: str
    feature_selector: str | None
    model_name: str
    model_params: dict[str, object]
    random_seed: int
```

### FidelitySpec

```python
class FidelitySpec:
    level: int
    sample_fraction: float
    n_folds: int
    max_iterations: int | None
    seeds: list[int]
```

### TrialResult

```python
class TrialResult:
    trial_id: str
    family: str
    pipeline_spec: PipelineSpec
    fidelity: FidelitySpec
    status: str
    primary_metric: str
    mean_score: float | None
    std_score: float | None
    fold_scores: list[float]
    fit_seconds: float
    predict_seconds: float
    peak_memory_mb: float | None
    failure_type: str | None
    failure_message: str | None
    started_at: str
    finished_at: str
```

### RunManifest

Le manifeste doit inclure :

- version du manifeste ;
- version du package ;
- configuration ;
- hashes des données ;
- profil ;
- plan de validation ;
- versions des dépendances ;
- seed ;
- meilleur pipeline ;
- essais finalistes ;
- métriques ;
- chemins des artefacts ;
- état de validation du notebook.

## 4. Modules

### data

Responsabilités :

- lecture CSV ;
- validation des chemins ;
- détection de séparateur raisonnable ;
- normalisation des noms sans modification destructive ;
- calcul des hashes ;
- cohérence train/test ;
- séparation cible.

### profiling

Responsabilités :

- inférence des types ;
- statistiques descriptives ;
- valeurs manquantes ;
- cardinalités ;
- duplications ;
- landmarks ;
- suspicion de texte libre ;
- rapport de fuite.

### validation

Responsabilités :

- choix du splitter ;
- matérialisation reproductible des folds ;
- conservation des indices ;
- validation des contraintes groupe/temps ;
- export des folds.

### components

Responsabilités :

- adapters sklearn ;
- imputers ;
- encoders ;
- transformateurs dates ;
- selectors ;
- modèles.

### pipelines

Responsabilités :

- grammaire ;
- compatibilité ;
- materialization ;
- construction `Pipeline` et `ColumnTransformer` ;
- sérialisation.

### search

Responsabilités :

- génération des candidats ;
- Optuna par famille ;
- allocation adaptative ;
- multi-fidélité ;
- pruning ;
- budget ;
- promotion ;
- diversité.

### evaluation

Responsabilités :

- exécution fold par fold ;
- scoring ;
- isolation des erreurs ;
- mesure du temps ;
- mesure mémoire si disponible ;
- prédictions OOF ;
- agrégation.

### tracking

Responsabilités :

- SQLite ;
- schéma des runs ;
- schéma des trials ;
- checkpoints ;
- stockage atomique ;
- reprise ;
- migration de schéma.

### memory

Responsabilités :

- métacaractéristiques ;
- recherche de datasets similaires ;
- warm start ;
- export pour futur méta-ranker.

### reporting

Responsabilités :

- leaderboard ;
- rapport HTML ;
- notebook ;
- validation du notebook ;
- comparaison des prédictions.

### runtime

Responsabilités :

- exécution locale ;
- gestion `n_jobs` ;
- limite de temps ;
- signaux d’arrêt ;
- annulation propre ;
- future abstraction distribuée.

## 5. Interfaces des adapters

```python
class ModelAdapter(Protocol):
    name: str

    def supported_tasks(self) -> set[str]: ...
    def supports_missing_values(self) -> bool: ...
    def supports_native_categoricals(self) -> bool: ...
    def supports_sparse_input(self) -> bool: ...
    def is_available(self) -> bool: ...
    def is_compatible(
        self,
        profile: DatasetProfile,
        pipeline_spec: PipelineSpec,
    ) -> bool: ...
    def suggest_params(
        self,
        trial: optuna.Trial,
        profile: DatasetProfile,
        fidelity: FidelitySpec,
    ) -> dict[str, object]: ...
    def build(
        self,
        params: dict[str, object],
        task: str,
        random_seed: int,
        n_jobs: int,
    ) -> BaseEstimator: ...
```

## 6. Allocation adaptative

L’allocateur doit exposer :

```python
class FamilyAllocator:
    def register(self, families: list[str]) -> None: ...
    def select_next(self, state: SearchState) -> str: ...
    def update(self, result: TrialResult) -> None: ...
    def snapshot(self) -> dict[str, object]: ...
    def restore(self, snapshot: dict[str, object]) -> None: ...
```

Politique V1 recommandée :

- exploration initiale de toutes les familles ;
- score de priorité de type UCB ;
- pénalité coût ;
- bonus amélioration ;
- quota de diversité ;
- désactivation temporaire après échecs répétés ;
- réactivation possible.

## 7. FidelityScheduler

```python
class FidelityScheduler:
    def initial_level(self) -> FidelitySpec: ...
    def should_promote(
        self,
        candidate: TrialResult,
        peers: list[TrialResult],
    ) -> bool: ...
    def next_level(self, current: FidelitySpec) -> FidelitySpec | None: ...
    def snapshot(self) -> dict[str, object]: ...
```

## 8. Stockage

Tables minimales :

```text
runs
datasets
profiles
validation_plans
pipeline_specs
trials
fold_results
artifacts
scheduler_state
events
```

Les écritures doivent être transactionnelles.

## 9. Reprise

La reprise doit :

1. lire le manifeste ;
2. vérifier les hashes des données ;
3. restaurer l’étude Optuna ;
4. restaurer l’allocateur ;
5. restaurer le scheduler ;
6. identifier les essais interrompus ;
7. marquer ou relancer uniquement les essais incomplets ;
8. respecter le budget restant ou un nouveau budget explicite.

## 10. Notebook

Le notebook est compilé à partir de templates.

Le code n’est pas rédigé librement par un LLM.

Les cellules doivent importer le package installé et reconstruire le pipeline à partir du `PipelineSpec` ou charger l’artefact final selon la section.

Le notebook est ensuite exécuté par `nbclient`.

## 11. Gestion des erreurs

Catégories minimales :

- `DataValidationError`
- `ConfigurationError`
- `IncompatiblePipelineError`
- `TrialTimeoutError`
- `TrialMemoryError`
- `ModelUnavailableError`
- `MetricError`
- `ArtifactValidationError`
- `ResumeError`

Chaque erreur d’essai doit être enregistrée sans arrêter le run global, sauf corruption du registre ou données invalides.
