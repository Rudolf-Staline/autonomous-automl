# Plan d’implémentation

## M0 — Bootstrap du dépôt

Livrables :

- `pyproject.toml`
- structure `src/`
- structure `tests/`
- configuration Ruff
- configuration Pyright
- configuration Pytest
- package importable
- CLI `automl --help`
- script qualité
- CI locale ou GitHub Actions facultative

Critères :

- `uv sync` fonctionne ;
- un test minimal passe ;
- Ruff et Pyright sont exécutables.

## M1 — Contrats et configuration

Livrables :

- `AutoMLConfig`
- `DatasetBundle`
- `DatasetProfile`
- `ValidationPlan`
- `PipelineSpec`
- `FidelitySpec`
- `TrialResult`
- `RunManifest`
- sérialisation JSON
- versionnement des schémas

Tests :

- round-trip JSON ;
- validation des valeurs ;
- compatibilité ascendante minimale ;
- refus des configurations invalides.

## M2 — Chargement et profilage

Livrables :

- lecture train/test ;
- séparation cible ;
- hashes ;
- cohérence des colonnes ;
- inférence des types ;
- missingness ;
- cardinalité ;
- constantes ;
- identifiants probables ;
- doublons ;
- landmarks simples ;
- export du profil.

Tests :

- CSV numériques ;
- CSV mixtes ;
- séparateurs ;
- cible absente ;
- train/test incohérents ;
- colonne entièrement manquante ;
- catégorie inconnue.

## M3 — Détection de fuite et validation

Livrables :

- `LeakageDetector`
- `ValidationPlanner`
- folds reproductibles ;
- support KFold, StratifiedKFold, GroupKFold ;
- support temporel explicite minimal ;
- export des folds ;
- rapport de justification.

Tests :

- colonne égale à la cible ;
- identifiant ;
- doublons ;
- groupe ;
- ordre temporel ;
- reproductibilité des folds.

## M4 — Registre de composants et construction des pipelines

Livrables :

- adapters modèles ;
- imputers ;
- encoders ;
- date transformer ;
- registry ;
- grammaire conditionnelle ;
- règles de compatibilité ;
- `build_pipeline(spec, profile)`.

Modèles noyau :

- Dummy ;
- LogisticRegression ;
- Ridge ;
- ElasticNet ;
- RandomForest ;
- ExtraTrees ;
- HistGradientBoosting.

Tests :

- construction déterministe ;
- clone sklearn ;
- catégories inconnues ;
- NaN ;
- sparse/dense ;
- sérialisation ;
- compatibilité filtrée.

## M5 — Évaluation cross-validée

Livrables :

- `PipelineEvaluator`
- scoring ;
- OOF ;
- isolation des erreurs ;
- mesure du temps ;
- agrégation ;
- gestion classification/régression ;
- journalisation.

Tests :

- aucune fuite d’imputation ;
- fold scores ;
- échec d’un modèle ;
- métrique incompatible ;
- OOF complet ;
- seed reproductible.

## M6 — Tracking et reprise

Livrables :

- SQLite ;
- migrations ;
- runs ;
- trials ;
- fold results ;
- events ;
- artifacts ;
- écritures atomiques ;
- reprise après interruption.

Tests :

- interruption simulée ;
- reprise ;
- aucun doublon d’essai ;
- corruption détectée ;
- hash dataset différent refusé.

## M7 — Recherche Optuna conditionnelle

Livrables :

- un optimizer par famille ;
- espaces conditionnels ;
- ask/tell ;
- baselines ;
- pruning ;
- limites par essai ;
- gestion modèles optionnels.

Tests :

- espace valide ;
- paramètres reproductibles ;
- essai pruned ;
- essai échoué ;
- étude persistante ;
- budget court.

## M8 — Multi-fidélité

Livrables :

- niveaux de fidélité ;
- sous-échantillonnage reproductible ;
- promotion ;
- confirmation ;
- diversité ;
- gestion du budget réservé.

Tests :

- passage faible vers moyen ;
- non-promotion ;
- diversité ;
- même sous-échantillon avec même seed ;
- budget final réservé.

## M9 — Allocation adaptative

Livrables :

- `FamilyAllocator`
- exploration initiale ;
- UCB ou stratégie équivalente ;
- pénalité coût ;
- bonus amélioration ;
- désactivation temporaire ;
- snapshot/restore.

Tests :

- toutes les familles explorées ;
- famille prometteuse priorisée ;
- famille coûteuse pénalisée ;
- famille en échec suspendue ;
- état restauré.

## M10 — Sélection et entraînement final

Livrables :

- leaderboard ;
- front de Pareto simple ;
- profils accuracy/balanced/fast ;
- réévaluation finalistes ;
- entraînement final ;
- pipeline sérialisé ;
- prédictions test.

Tests :

- baseline battue sur datasets contrôlés ;
- sélection cohérente ;
- prédictions stables ;
- artefact rechargeables.

## M11 — Mémoire et warm start

Livrables :

- métacaractéristiques ;
- stockage inter-runs ;
- similarité ;
- récupération de pipelines historiques ;
- injection warm start ;
- fallback sans historique.

Tests :

- dataset similaire ;
- dataset sans voisin ;
- ordre warm start ;
- aucune fuite entre datasets.

## M12 — Notebook et rapport

Livrables :

- manifeste final ;
- compilateur notebook ;
- rapport HTML ;
- exécution nbclient ;
- comparaison prédictions ;
- validation des artefacts.

Tests :

- notebook exécutable ;
- aucune cellule manuelle requise ;
- prédictions correspondantes ;
- erreurs explicites ;
- rapport généré.

## M13 — CLI complet et documentation

Livrables :

- `fit`
- `resume`
- `inspect`
- `leaderboard`
- `export-notebook`
- `validate-artifacts`
- `doctor`
- README final
- exemples
- guide développeur
- troubleshooting

Tests :

- smoke tests CLI ;
- codes de sortie ;
- messages d’erreur ;
- chemins relatifs ;
- Windows/Linux best effort.

## M14 — Benchmark initial

Livrables :

- harness de benchmark ;
- random search ;
- Optuna simple ;
- scheduler adaptatif ;
- scheduler + warm start ;
- métriques anytime ;
- rapport comparatif.

Le benchmark lourd ne bloque pas les tests standards.

## Politique de milestone

Un milestone est terminé uniquement si :

- le code est intégré au flux principal ;
- les tests spécifiques passent ;
- les tests antérieurs passent ;
- la documentation est mise à jour ;
- `IMPLEMENTATION_STATUS.md` est mis à jour.
