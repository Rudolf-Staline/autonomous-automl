# Décisions techniques

## TD-001 — Langage et packaging

- Python 3.12
- gestion avec `uv`
- package en structure `src/`
- `pyproject.toml` comme source de vérité

## TD-002 — DataFrame

Choix initial recommandé : pandas.

Raisons :

- compatibilité maximale avec scikit-learn ;
- simplicité ;
- écosystème mature ;
- sérialisation familière.

Polars peut être utilisé pour le profilage ou la lecture si cela apporte un gain mesuré, mais les frontières avec scikit-learn doivent rester explicites.

## TD-003 — Validation et pipelines

- scikit-learn `Pipeline`
- scikit-learn `ColumnTransformer`
- `clone` pour chaque fold
- aucun fit de transformateur avant le split

## TD-004 — Optimisation

- Optuna pour la suggestion des hyperparamètres
- une étude ou un espace logique par famille
- API ask/tell lorsque nécessaire
- stockage persistant SQLite

Optuna ne contrôle pas seul toute l’allocation. Le contrôleur adaptatif reste une couche propre au projet.

## TD-005 — CLI

Typer.

Commandes prévues :

```text
fit
resume
inspect
leaderboard
export-notebook
validate-artifacts
doctor
```

## TD-006 — Sérialisation

- JSON pour les contrats et manifestes
- Parquet pour les historiques tabulaires si disponible
- CSV pour les sorties utilisateur simples
- Joblib pour le pipeline final
- SQLite pour l’état transactionnel

## TD-007 — Notebook

- `nbformat`
- `nbclient`
- cellules générées par templates déterministes
- exécution automatique
- validation des outputs

## TD-008 — Rapports

Le rapport HTML peut être généré avec Jinja2.

Aucune interface web interactive dans la V1.

## TD-009 — Logging

- logs console lisibles
- logs fichier structurés JSONL
- niveaux DEBUG, INFO, WARNING, ERROR
- identifiants de run et trial dans chaque événement

## TD-010 — Configuration

La configuration doit être :

- validée ;
- sérialisable ;
- versionnée ;
- modifiable par CLI ;
- sauvegardée dans le manifeste.

Pydantic peut être utilisé pour les contrats d’entrée et les manifestes.

## TD-011 — Modèles optionnels

XGBoost, LightGBM et CatBoost sont des extras.

Exemple :

```bash
uv sync --extra boosting
```

Le noyau doit fonctionner sans eux.

## TD-012 — Parallélisme

V1 :

- parallélisme local ;
- contrôle du nombre de workers ;
- prévention du sur-abonnement ;
- un seul niveau principal de parallélisme à la fois.

Éviter :

```text
Optuna workers × CV folds × model n_jobs
```

sans contrôle, car cela peut saturer la machine.

## TD-013 — Temps et budget

Le budget est global.

Le moteur doit réserver du temps pour :

- confirmation des finalistes ;
- entraînement final ;
- génération du notebook ;
- validation des artefacts.

Le search controller ne doit pas consommer 100 % du budget.

## TD-014 — Métriques internes

Toutes les métriques sont converties vers une convention interne “plus grand = meilleur”.

Exemples :

- RMSE interne = `-rmse`
- MAE interne = `-mae`

Le rapport conserve la valeur utilisateur originale.

## TD-015 — Reproductibilité

- seed globale ;
- seeds dérivées par run, famille, trial et fold ;
- folds matérialisés ;
- versions des dépendances enregistrées ;
- déterminisme best effort documenté pour les modèles non strictement déterministes.

## TD-016 — Compatibilité

Les règles de compatibilité sont des fonctions testables, pas des conditions dispersées.

## TD-017 — Sélection finale

La sélection finale ne dépend pas uniquement du meilleur score brut.

Le système conserve :

- score moyen ;
- écart-type ;
- coût ;
- latence ;
- mémoire ;
- taux d’échec ;
- niveau de fidélité ;
- nombre de seeds.

La politique dépend du profil :

- `accuracy`
- `balanced`
- `fast`

## TD-018 — Pas de LLM dans le chemin critique

Un LLM peut éventuellement produire du texte explicatif dans une version ultérieure.

Il ne doit pas :

- inventer le code du pipeline final ;
- décider seul du meilleur modèle ;
- modifier les résultats ;
- générer une logique non testée dans le notebook.
