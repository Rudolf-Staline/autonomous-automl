# État d’implémentation

Ce fichier est le journal de réalisation de la V1. Il est mis à jour après chaque
milestone et chaque campagne de validation.

## Statut global

- État : audit adversarial de la release candidate en cours
- Périmètre gelé : fonctionnalités M0 à M9, plus le strict nécessaire au parcours complet
- Chantier actif : hygiène Git, exactitude CLI/rapport, documentation et preuve par clone neuf
- Dernière mise à jour : 2026-07-18
- Dernière gate complète connue : 393 tests réussis ; nouvelle gate RC à rejouer

## Audit adversarial RC — en cours au 2026-07-18

- Le noyau M0–M9 est gelé : aucune fonctionnalité produit supplémentaire n'est engagée.
- Bloqueur découvert : `HEAD` ne contient encore que les documents de spécification ;
  l'implémentation est présente dans l'arbre de travail mais non suivie. Un commit RC local
  est requis avant qu'un véritable `git clone` puisse constituer une preuve d'installation.
- Recherche de secrets et d'hygiène : aucun secret réel, fichier `.env`, chemin personnel,
  journal, base SQLite, modèle Joblib ou artefact lourd détecté dans les sources à livrer.
- Démonstration de référence : PASS en 18,91 s, trois scénarios, environ 225 MiB de RSS
  maximal, 47 fichiers stables produits sous le répertoire de sortie.
- Reprise adversariale : PASS ; deux essais terminés avant interruption sont conservés avec
  leur identifiant et `attempt_count=1`, puis le budget effectif passe de 8 s à 11 s.
- Corruption volontaire du modèle : PASS ; taille/SHA-256 invalides détectés avant chargement.
- Reconstruction depuis le `PipelineSpec` et rejeu des prédictions : PASS, 36/36 prédictions
  strictement identiques sur le scénario de classification.
- Écarts de release en correction : sortie CLI trop verbeuse, preuve de neutralisation peu
  visible dans le rapport, guide juge trop long, ordre du README et version du changelog.

Les mesures et résultats définitifs remplaceront ces observations intermédiaires après les
gates locales puis un clone Git neuf créé à partir du commit RC.

### Gate locale RC après corrections

Exécution séquentielle réelle le 2026-07-18 :

| Commande | Statut | Durée murale | Résultat / avertissement |
|---|---:|---:|---|
| `uv sync --frozen` | PASS | 0,02 s | 74 paquets contrôlés ; aucun avertissement |
| `uv run pytest` | PASS | 24,20 s | 394/394 en 23,39 s pytest ; aucun avertissement |
| `uv run ruff check .` | PASS | 0,04 s | aucun diagnostic |
| `uv run ruff format --check .` | PASS | 0,03 s | 98 fichiers conformes |
| `uv run pyright` | PASS | 9,72 s | 0 erreur, 0 avertissement |
| `uv run automl --help` | PASS | 1,36 s | huit commandes publiques affichées |
| `uv run automl demo --output-root runs/rc-local-demo` | PASS | 19,53 s | trois validations PASS ; pic RSS 229 784 KiB |

La démo locale finale a produit 47 fichiers stables (1 868 259 octets) : 16 pour
classification, 16 pour régression et 15 pour fuite. Les résultats observés sont
ROC AUC 0,835926, RMSE 8,180915 et ROC AUC post-neutralisation 0,837941. Un timeout
de pipeline individuel a été conservé dans le run fuite ; il n'a pas arrêté le run.

### Commandes des attaques de reprise et de corruption

Reprise exécutée :

```bash
uv run automl fit examples/classification_train.csv \
  --config examples/classification_config.json \
  --output runs/adversarial-resume \
  --budget 8s \
  --interrupt-after-trials 2
uv run automl resume runs/adversarial-resume --additional-budget 3s
```

Avant reprise : 2 essais terminés, budget consommé 2,376211 s sur 8 s et statut
`interrupted`. Après reprise : 14 essais, budget consommé 7,466776 s sur 11 s ; les
deux IDs initiaux restent uniques avec `attempt_count=1`.

Corruption exécutée sur une copie hors dépôt :

```bash
cp -a runs/adversarial-baseline/classification-resumed /tmp/tmp.vVu89kyz1y/run
printf CORRUPTED >> /tmp/tmp.vVu89kyz1y/run/best_pipeline.joblib
uv run automl validate-artifacts /tmp/tmp.vVu89kyz1y/run
```

La dernière commande sort avec le code 1 et `ArtifactValidationError: artifact size
mismatch: best_pipeline`. Un appel direct à `load_best_pipeline` reçoit la même
erreur avant désérialisation. Le `PipelineSpec` du manifeste a aussi été passé à
`build_pipeline`, ajusté sur le train puis appliqué au test : les 36 prédictions
étaient strictement identiques au fichier persisté. Le test automatisé
`test_joblib_checksum_is_verified_before_deserialization` garantit en plus que
`joblib.load` n'est jamais appelé après une modification taille/hash.

## Gate Build Week précédente — historique

| Critère de livraison | Résultat vérifié |
|---|---|
| `uv sync --frozen` | PASS — 74 paquets contrôlés |
| Tests unitaires/intégration/fuite/reprise | PASS — 393/393 |
| `ruff check .` | PASS |
| `ruff format --check .` | PASS — 98 fichiers |
| `pyright` | PASS — 0 erreur, 0 avertissement |
| CLI `--help` et commandes publiques | PASS |
| Classification avec test et catégorie inconnue | PASS |
| Régression avec prédictions test | PASS |
| Copie de cible et identifiant suspect | PASS — exclus avant recherche |
| Interruption puis reprise sans perte d'essais | PASS |
| Chargement Joblib vérifié par SQLite/SHA-256 | PASS |
| Cohérence des prédictions reproduites | PASS |
| Rapport HTML issu des artefacts persistés | PASS |
| Générateur d'exemples déterministe | PASS — hashes identiques après régénération |
| Installation depuis un clone temporaire neuf | PASS — nouvelle `.venv`, CLI et demo complète |
| Démonstration `uv run automl demo` | PASS — trois scénarios en environ 17 s |
| Notebook certifié | REPORTÉ explicitement hors périmètre Build Week |
| Docker | Dockerfile fourni ; build non exécuté car aucun moteur Docker/Podman n'est installé |

La procédure primaire demandée aux juges, `uv sync --frozen && uv run automl demo`,
a été vérifiée depuis une copie versionnée temporaire puis un clone neuf. Elle ne
dépend ni de Docker ni d'un service externe.

## État fonctionnel audité — 2026-07-18

### Fonctionnalités réellement terminées

- contrats JSON stricts et versionnés pour configuration, données, profil, validation,
  pipelines, fidélités, essais, manifeste et résultat public ;
- chargement CSV mono/multi-fichier, validation du schéma et séparation stricte du
  fichier test ;
- profilage train-only, inférence de tâche, détection explicable de copies de cible,
  corrélations suspectes, identifiants et autres risques de fuite ;
- plans StratifiedKFold/KFold/groupe/temps/folds prédéfinis matérialisés et audités ;
- pipelines scikit-learn fold-safe avec imputations, encodages inconnus-safe,
  transformations et registre de modèles obligatoires/optionnels ;
- évaluation CV, OOF, métriques orientées, seeds déterministes, limites de temps et
  confinement des erreurs individuelles ;
- registre SQLite transactionnel, artefacts atomiques vérifiés, leases, fencing,
  snapshots et reprise des essais interrompus ;
- optimisation Optuna persistante, recherche à quatre fidélités, promotions et
  allocation adaptative coût-aware réellement connectées par `SearchController` ;
- composants testés de leaderboard, sélection Pareto/profils, confirmation des
  finalistes et réentraînement final depuis le `PipelineSpec` ;
- façade publique `AutoMLRun.fit/resume` et CLI complète jusqu'aux prédictions,
  validation d'artefacts et démonstration en une commande ;
- rapport HTML autonome compilé depuis le manifeste, SQLite et l'inventaire réel.

### Fonctionnalités partielles ou explicitement reportées

- le notebook certifié n'est pas généré et `RunResult.notebook_path` reste `None` ;
- M11 (mémoire inter-datasets) et M14 (benchmarks comparatifs lourds) sont reportés ;
- les boosters optionnels sont découverts paresseusement mais ne font pas partie de
  l'installation ni de la démonstration noyau ;
- le Dockerfile est fourni mais son image n'a pas été construite sur cet hôte, qui
  ne possède aucun moteur Docker/Podman/Buildah.

### Commandes publiques exécutables

```text
uv run automl --help
uv run automl fit
uv run automl resume
uv run automl inspect
uv run automl leaderboard
uv run automl predict
uv run automl validate-artifacts
uv run automl demo
uv run automl doctor
uv run python scripts/quality.py
```

### Bloqueurs de release candidate restant à cet instant

1. ajouter l'implémentation complète au suivi Git et produire un commit RC local ;
2. rejouer les cinq gates finales après les corrections de présentation ;
3. cloner ce commit dans un répertoire neuf avec un cache et une `.venv` neufs ;
4. exécuter dans ce clone l'installation, l'aide CLI, la démo et les commandes juge.

### Périmètre reporté après le hackathon

- M11 mémoire inter-datasets et méta-ranker ;
- M14 benchmark lourd et comparaison étendue aux autres frameworks AutoML ;
- interface web, parallélisme distribué et ensembles avancés ;
- support enrichi des séries temporelles, NLP et formats autres que CSV ;
- notebook certifié si son intégration met en risque le parcours principal ;
- dépendances boosting optionnelles dans le scénario de démonstration.

Le rapport HTML, la sélection finale, les prédictions et les commandes publiques ne
sont pas considérés comme de nouveaux milestones fonctionnels : ils ferment le
parcours M0–M9 nécessaire à une livraison testable.

## Architecture reformulée

Le moteur suit un flux unique et traçable : chargement et hash des CSV, profilage,
détection des risques de fuite, matérialisation d’un plan de validation, génération
de `PipelineSpec` compatibles, recherche adaptative multi-fidélité, évaluation fold
par fold, sélection et réentraînement, puis compilation et validation des artefacts.

Trois couches restent strictement séparées :

1. les contrats JSON versionnés décrivent configuration, données, validation,
   pipelines, fidélité, essais et manifeste ;
2. des factories déterministes construisent les objets scikit-learn à partir de ces
   descriptions ;
3. les services exécutent profilage, validation, recherche, évaluation, stockage,
   reprise et reporting.

`PipelineSpec` est la source de vérité d’un pipeline, jamais l’estimateur sérialisé.
L’imputation, l’encodage et toute transformation apprise sont contenus dans la
`Pipeline` évaluée à l’intérieur de chaque fold. Les folds sont matérialisés et
enregistrés. Le test éventuel reste hors de la recherche. Optuna suggère les
hyperparamètres, tandis qu’un contrôleur propre au projet choisit les familles,
fidélités et promotions sous budget. SQLite conserve transactionnellement les runs,
essais (y compris les échecs), folds, événements, artefacts et snapshots nécessaires
à la reprise. Le manifeste relie tous les artefacts ; le notebook est compilé depuis
ce manifeste dans la cible d'architecture initiale, mais cette branche est reportée
pour Build Week. La livraison vérifie à la place le pipeline Joblib enregistré et
compare directement ses prédictions aux prédictions persistées ; le rapport HTML
est compilé depuis le manifeste et SQLite.

## Plan courant

Le périmètre fonctionnel est gelé après M9. Le travail restant suit uniquement le
chemin de livraison Build Week :

1. **Terminé** — auditer le dépôt et figer M11/M14 ;
2. **Terminé** — connecter `AutoMLRun.fit/resume`, sélection finale, Joblib,
   prédictions, manifeste et vérification de replay ;
3. **Terminé** — livrer la CLI Rich, le rapport HTML persistant et la démonstration
   classification/régression/fuite en une commande ;
4. **Terminé** — README, guides juge/vidéo/Devpost, Dockerfile et historique Codex ;
5. **En cours** — gates complètes, commit RC, installation et demo depuis un vrai
   clone Git propre.

## Plan de fichiers

```text
pyproject.toml
src/autonomous_automl/
├── __init__.py                 # API publique AutoMLRun / AutoMLConfig
├── api/run.py                  # orchestration fit/resume
├── cli/app.py                  # commandes Typer
├── contracts/                  # modèles JSON versionnés
├── data/                       # CSV, validation, hashes, DatasetBundle
├── profiling/                  # types, statistiques, fuite
├── validation/                 # planification et matérialisation des folds
├── components/                 # transformateurs et adapters modèles
├── pipelines/                  # grammaire, compatibilité, factories
├── evaluation/                 # métriques, CV, OOF, isolation des erreurs
├── tracking/                   # schéma SQLite, transactions, artefacts, reprise
├── search/                     # Optuna, budget, fidélité, UCB, contrôleur
├── reporting/                  # rapport HTML issu des artefacts persistés
├── runtime/                    # budget, seeds, arrêt propre
└── utils/                      # JSON atomique, hashes, logs structurés
tests/
├── unit/
├── integration/
├── leakage/
├── reproducibility/
├── resume/
├── cli/
├── notebook/
└── benchmarks/
examples/                       # petits CSV synthétiques reproductibles
docs/                           # architecture, utilisation, développement, dépannage
scripts/quality.py              # commande de qualité agrégée
scripts/generate_examples.py    # génération déterministe des exemples
```

## Milestones

| Milestone | Statut | Notes |
|---|---|---|
| M0 — Bootstrap | Terminé | Python 3.12, uv, package, CLI et qualité validés |
| M1 — Contrats | Terminé | 9 contrats centraux, JSON strict, 113 tests dédiés |
| M2 — Données et profilage | Terminé | CSV multi-source, firewall cible et profil train-only validés |
| M3 — Fuite et validation | Terminé | Détection explicable, folds persistés et audit sécurité |
| M4 — Pipelines | Terminé | Composants fold-safe, registry, compatibilité et reconstruction |
| M5 — Évaluation | Terminé | CV fold-local, métriques orientées, OOF et échecs confinés |
| M6 — Tracking et reprise | Terminé | SQLite transactionnel, artefacts vérifiés, lease et reprise |
| M7 — Optuna | Terminé | Études par modèle, ask/tell/reconcile déterministes et persistants |
| M8 — Multi-fidélité | Terminé | Quatre niveaux, sampling fold-safe, promotions et réserves |
| M9 — Allocation adaptative | Terminé | UCB coût-aware et contrôleur crash-safe connectés |
| M10 — Sélection finale | Terminé | Leaderboard, confirmation, entraînement et replay Joblib intégrés |
| M11 — Mémoire | Reporté | Hors parcours de démonstration Build Week |
| M12 — Notebook et rapport | Partiel gelé | Rapport HTML terminé ; notebook certifié reporté |
| M13 — CLI et documentation | Terminé | CLI, demo, README, guides juge/vidéo/Devpost et Dockerfile |
| M14 — Benchmark | Reporté | Harness lourd hors deadline ; exemples courts conservés |

## Décisions prises

- pandas est retenu pour la V1 afin de conserver une frontière directe et testable
  avec scikit-learn.
- Pydantic v2 portera les contrats persistants validés et leur sérialisation JSON.
- Les dépendances boosting resteront des extras détectés dynamiquement derrière des
  adapters ; aucune importation optionnelle ne sera faite au démarrage du noyau.
- Les historiques utilisateur auront toujours un fallback CSV ; Parquet ne sera
  utilisé que si son moteur optionnel est disponible.
- Le parallélisme principal restera au niveau du modèle (`n_jobs` contrôlé), sans
  multiplier simultanément workers Optuna, folds et threads internes.
- Les écritures d’artefacts JSON et Joblib seront atomiques ; Joblib ne chargera que
  les artefacts produits dans le répertoire de run.
- `DatasetBundle` est le descripteur persistant (sources, hashes et schéma) ; les
  DataFrames brutes vivront uniquement dans un `LoadedDataset` d’exécution non
  sérialisé afin de respecter la séparation description/exécution.
- Le target encoding est volontairement absent tant qu’un encodeur cross-fitté et
  ses tests de fuite dédiés ne sont pas disponibles ; one-hot, ordinal et fréquence
  couvriront la V1 sans introduire d’encodeur cible naïf.
- Le hash du profil combine uniquement les sources d’entraînement ; un changement
  du fichier test ne peut donc modifier ni profil, ni plan, ni recherche.
- Les colonnes ID, groupe et fold prédéfini sont sorties du feature matrix dès le
  chargement et conservées dans des séries runtime dédiées ; la colonne temporelle
  reste une feature connue tout en étant copiée pour planifier le split.
- Les copies exactes/quasi exactes de cible, corrélations de régression quasi
  parfaites et IDs haute confiance sont exclus uniquement avec un `LeakageFinding`
  explicite ; les noms suspects et doublons restent des avertissements auditables.
- Une seule stratégie spéciale parmi groupe, temps et fold prédéfini est acceptée
  en V1. Les folds sont stockés par positions ; `ValidationAudit` persiste les
  contrôles de bornes, intersections, groupes, temps, couverture et doublons.
- Les catégories sont normalisées en chaînes dans le pipeline avant imputation ;
  one-hot ignore les inconnues, ordinal réserve des codes inconnus et fréquence
  apprend ses mappings dans le fold avec fallback zéro.
- Les infinis numériques deviennent des valeurs manquantes dans le pipeline. HGB
  reçoit toujours une matrice dense ; les modèles compatibles conservent le sparse.
- Le registry noyau contient sept modèles obligatoires. XGBoost, LightGBM et
  CatBoost ne sont jamais importés au démarrage et leur absence est rapportée.
- Les métriques internes suivent toujours « plus grand est meilleur » ; RMSE, MAE
  et log-loss sont donc négatives pour l'optimiseur, tandis que la valeur positive
  d'origine est persistée dans chaque `FoldResult`.
- Les probabilités de classification sont réalignées sur un ordre global et
  déterministe des classes avant scoring et accumulation OOF. Les seeds d'une
  fidélité sont dérivées par essai et fold avec SHA-256, sans utiliser le hash
  aléatoire de Python.
- Lorsqu'un timeout est configuré, chaque fold s'exécute dans un processus
  annulable (`forkserver`/`spawn`, avec fallback interactif) ; un appel natif bloqué
  peut donc être terminé et conservé comme `TrialTimeoutError` sans bloquer le run.
- SQLite est la source de vérité de l'état mutable entre checkpoints ; le manifeste
  reste l'inventaire lisible et versionné. Essai, folds, snapshots des composants,
  budget consommé et révision sont commités dans une transaction `BEGIN IMMEDIATE`.
- L'identité d'un candidat est le SHA-256 canonique de son `PipelineSpec` et de sa
  `FidelitySpec`. Une réservation concurrente n'a qu'un gagnant ; une réservation
  terminée n'est jamais rejouée et une réservation interrompue reprend le même ID.
- Les chemins sources sont désormais absolus dans `DatasetBundle`, ce qui permet de
  revérifier exactement les hashes train et test même si `resume` est lancé depuis
  un autre répertoire de travail.
- Joblib n'est chargé qu'après confinement du chemin dans le run et vérification de
  la taille et du SHA-256. Les écritures Joblib utilisent temporaire, `fsync` et
  `os.replace` ; JSON/CSV utilisent les primitives atomiques communes.
- Chaque tentative relancée reçoit un token et chaque propriétaire de run un epoch
  de lease : un résultat tardif ne peut pas écraser la tentative active. Les
  snapshots restaurés sont des objets complets attachés à une révision unique.
- Les artefacts enregistrés sont immuables. Joblib est ouvert avec une traversée
  confinée sans symlink, hashé puis désérialisé depuis le même descripteur, et sa
  provenance est vérifiée dans SQLite avant le chargement.
- Optuna utilise une étude persistante par modèle sous une famille logique. Un
  nouveau `TPESampler` reçoit une seed dérivée du numéro d'essai et de l'historique,
  ce qui rend la suggestion suivante identique après réouverture de l'étude.
- Les fidélités faible, moyenne, complète et confirmation sont des contrats
  déterministes adaptés au nombre réel de folds. Une promotion conserve strictement
  le même `PipelineSpec` et ne change que la `FidelitySpec`.
- Le sous-échantillonnage de fidélité ne lit que le côté entraînement du fold : il
  est stratifié pour la classification, sélectionne des groupes entiers, et prend
  un préfixe chronologique pour une validation temporelle. Le fichier test n'est
  jamais consulté par ce service.
- Le parcours public utilise une horloge monotone et protège 25 % pour la
  confirmation puis 20 % pour finalisation, artefacts et replay. Son snapshot persiste seulement
  le temps déjà consommé afin que la durée d'arrêt ne soit pas facturée à la reprise.
- `FamilyAllocator` garantit une exploration initiale de chaque famille, puis
  combine performance, amélioration, coût, incertitude et diversité. Une série
  d'échecs suspend temporairement une famille sans la supprimer définitivement.
- `SearchController` choisit réellement les familles via l'allocateur, demande les
  paramètres aux études Optuna, exécute d'abord les promotions en attente et commit
  résultat, folds, budget et snapshots dans une transaction SQLite. Les erreurs
  d'un candidat deviennent un `TrialResult` échoué et la recherche continue.
- Une tentative interrompue est reconstruite depuis les contrats SQLite et son
  candidat Optuna persistant. Token de tentative et epoch de lease empêchent tout
  résultat tardif ; une promotion interrompue reprend la file du scheduler sans
  créer une nouvelle suggestion.
- `AutoMLRun` est la façade publique unique : elle enchaîne données, diagnostic,
  recherche, confirmation, sélection, réentraînement, vérification Joblib et
  compilation des artefacts sans dupliquer les composants M0–M9.
- Rich est utilisé uniquement à la frontière CLI. La source de vérité de la
  progression reste un flux JSON Lines sans données brutes utilisateur.
- Le rapport HTML est généré depuis le manifeste, le registre SQLite et
  l'inventaire d'artefacts persistés ; aucune statistique de présentation n'est
  recalculée depuis les CSV bruts.
- Sur les plateformes disposant de `forkserver`, le module du worker de fold est
  préchargé afin de préserver l'isolation des timeouts sans payer les imports
  pandas/scikit-learn à chaque fold court.
- Le notebook certifié est explicitement reporté : son absence est exposée par
  `RunResult.notebook_path=None` et ne sera pas masquée par un notebook décoratif.

## Risques identifiés

| Risque | Mitigation prévue |
|---|---|
| Fuite par prétraitement ou target encoding | Pipelines sklearn fittées par fold ; target encoding exclu tant qu’un encodeur cross-fitté n’est pas prouvé par tests |
| Fuite groupe/temps/doublons | Folds matérialisés, contrôles d’intersection et rapport de risques avant recherche |
| Budget dépassé par un essai long | réserve finale explicite, deadline globale, fidélités bornées et vérification avant chaque essai |
| Reprise incohérente | hashes de sources/configuration, transactions SQLite, identifiants déterministes, snapshots versionnés |
| Non-déterminisme de modèles | hiérarchie de seeds, folds persistés, `n_jobs` borné, tolérance documentée |
| Explosion dense du one-hot | estimation de cardinalité, compatibilité sparse/dense et filtrage avant construction |
| Dépendances optionnelles absentes | découverte paresseuse, disponibilité enregistrée, registre minimal toujours fonctionnel |
| Notebook dépendant du contexte local | fonctionnalité reportée plutôt que livrée sans preuve ; rapport HTML et replay Joblib couvrent la reproductibilité soumise |
| Python 3.12/uv absents sur l’hôte | installer `uv`, épingler `requires-python`, laisser `uv` provisionner Python 3.12 |
| Portabilité des manifestes | champs `*_version`, validation stricte et migrations explicites du registre |

## Tests exécutés

- Audit initial : dépôt Git propre, uniquement les documents de spécification.
- Audit environnement initial : Python système 3.14.4, `uv` et Python 3.12 absents.
- M0 : `uv sync` réussi avec uv 0.11.29 et CPython 3.12.13 géré par uv.
- M0 : `pytest` — 3 tests réussis.
- M0 : `ruff check .` — réussi.
- M0 : `ruff format --check .` — réussi après formatage initial.
- M0 : `pyright` — 0 erreur, 0 avertissement.
- M0 : `automl --help` et `automl doctor` — réussis.
- M1 : 113 tests de contrats et 8 tests d’utilitaires ; 124 tests globaux réussis.
- M1 : `ruff check .`, `ruff format --check .` et `pyright` — réussis.
- M2 : 51 nouveaux tests loader/profiler/acceptation ; 175 tests globaux réussis.
- M2 : `ruff check .`, `ruff format --check .` et `pyright` — réussis sans
  avertissement de dépréciation dans la suite.
- M3 : 27 nouveaux tests leakage/validation/reproductibilité ; 202 tests globaux
  réussis.
- M3 : `ruff check .`, `ruff format --check .` et `pyright` — réussis.
- M4 : 53 nouveaux tests composants/adapters/pipelines ; 255 tests globaux réussis.
- M4 : clone sklearn, catégories inconnues, sparse/dense, Joblib et reconstruction
  `PipelineSpec` validés ; Ruff, format et Pyright réussis.
- M5 : 15 tests d'acceptation dédiés (métriques auto, orientation des pertes,
  incompatibilité, alignement OOF, reproductibilité, fit et imputation fold-local,
  seed indépendante du tracking, probabilités invalides, erreur modèle et timeout
  réellement interruptible) ; 270 tests globaux réussis.
- M5 : `scripts/quality.py` — Ruff lint et format réussis, Pyright 0 erreur/0
  avertissement, Pytest 270 réussis.
- M6 : 25 tests registre/reprise/artefacts ; migrations idempotentes,
  round-trip completed/failed, rollback transactionnel, concurrence, lease,
  snapshots, budget, reprise d'un running abandonné, hashes et corruption validés.
- M6 : Ruff et format réussis, Pyright 0 erreur/0 avertissement, 290 tests globaux
  réussis.
- M7 : 14 tests Optuna ; espaces cœur compatibles/JSON, suggestions reproductibles,
  completed/failed/pruned, tell idempotent, identité dataset/templates, reprise RNG
  et réconciliation d'un résultat métier après crash validés.
- M7 : `scripts/quality.py` — Ruff et format réussis, Pyright 0 erreur/0
  avertissement, 309 tests globaux réussis.
- M8 : 34 tests dédiés ; quatre niveaux, promotion/non-promotion, score ajusté,
  diversité, état exact après reprise, échantillonnage reproductible, classes rares,
  groupes, temps, firewall du fichier test et réserve finale validés.
- M8 : `scripts/quality.py` — Ruff et format réussis, Pyright 0 erreur/0
  avertissement, 343 tests globaux réussis.
- M9 : 27 nouveaux tests allocateur/contrôleur/reprise Optuna ; exploration de
  toutes les familles, UCB, coûts, diversité, cooldown, snapshots, promotions,
  erreurs isolées, déduplication, retry low/promotion et budget ajouté validés.
- M9 : Ruff et format ciblés réussis, Pyright 0 erreur/0 avertissement ; la suite
  globale Pytest passe avec 384 tests (incluant les premiers composants M10).
- Audit Build Week initial : Ruff/format/Pyright réussis et 390 tests globaux
  réussis en 4,97 s avant l'intégration publique.
- Parcours API réels : classification ROC AUC avec catégorie inconnue, régression
  RMSE et dataset contenant copie de cible/identifiant ; modèles, prédictions et
  rapports produits puis rechargés.
- Reprise publique réelle : checkpoint après deux essais, mêmes IDs conservés,
  budget additionnel appliqué et run terminé avec prédictions rejouées.
- CLI réelle : `--help`, `fit`, `inspect`, `leaderboard`, `predict` et
  `validate-artifacts` exécutés sur les artefacts produits.
- Démonstration : `uv run automl demo --output-root runs/cli-demo-preload` réussie
  en environ 17 s ; classification 0,791759 ROC AUC, régression 8,180915 RMSE,
  fuite 0,837941 ROC AUC, checks d'artefacts `PASS`.
- Trois tests d'intégration Build Week supplémentaires réussis en 18,82 s : fit et
  replay, interruption/reprise, régression et neutralisation de fuite.
- Reproductibilité des exemples : exécution de `scripts/generate_examples.py`, puis
  comparaison SHA-256 avant/après ; huit fichiers CSV/JSON strictement identiques.
- Clone propre : snapshot courant commité dans un dépôt temporaire, cloné dans un
  autre répertoire sans `.venv`, puis `uv sync --frozen`, `automl --help` et
  `automl demo` réussis. Les trois scénarios et leurs validations d'artefacts ont
  affiché `PASS`.
- Gate finale après correction de la compatibilité `doctor` : `uv sync --frozen`,
  393 tests en 23,03 s, Ruff lint, Ruff format (98 fichiers) et Pyright (0/0), tous
  réussis dans l'ordre demandé.

## Échecs connus

- Un budget manuel de 4 s par scénario est insuffisant avant préchauffage pour la
  démonstration multi-processus ; la commande refuse désormais moins de 6 s.
- Un essai peut légitimement finir `failed` à la frontière du budget. Il reste dans
  SQLite et le run continue tant qu'un pipeline utilisable, au minimum la baseline,
  a réussi.
- Le notebook de reproduction n'est pas livré dans le périmètre Build Week ; la
  reproductibilité vérifiée porte sur le manifeste, le `PipelineSpec`, Joblib et les
  prédictions enregistrées.
- Le Dockerfile suit l'image officielle uv/Python 3.12 épinglée et sa commande
  `uv sync --frozen --no-editable` a été validée en dry-run. Le build d'image n'a
  pas pu être exécuté : les binaires Docker, Podman et Buildah sont absents de
  l'hôte. La procédure sans Docker est la voie primaire vérifiée.

## Prochaines actions

Les actions suivantes sont post-hackathon et n'appartiennent pas à la livraison :

1. certifier la génération/exécution du notebook depuis le manifeste ;
2. ajouter la mémoire inter-datasets de M11 ;
3. exécuter les benchmarks comparatifs lourds de M14 ;
4. construire l'image Docker sur une machine disposant du daemon, sans remplacer la
   procédure `uv` déjà vérifiée.
