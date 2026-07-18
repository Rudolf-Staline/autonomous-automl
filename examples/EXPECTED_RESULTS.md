# Résultats essentiels attendus

`uv run automl demo` utilise une seed fixe et trois petits datasets générés par
`scripts/generate_examples.py`. Les scores précis peuvent légèrement varier selon
la version de scikit-learn et les performances de la machine ; les invariants
suivants doivent toujours être vrais :

- les trois runs finissent avec le statut `completed` ;
- le run de classification sélectionne `roc_auc` et écrit 36 prédictions ;
- le run de régression utilise `rmse`, produit un score fini et écrit 36
  prédictions ;
- le scénario de fuite exclut au minimum `customer_id` et `target_copy` avant la
  recherche (il exclut aussi `approved_after_review`, copie exacte de la cible) ;
- le scénario classification est d'abord interrompu après deux essais, puis repris
  sans suppression de ces essais ;
- chaque run terminé contient `best_pipeline.joblib`, `best_pipeline_spec.json`,
  `manifest.json`, `leaderboard.csv`, `trials.csv`, `report.html` et son registre
  SQLite ;
- la validation finale affiche `PASS` pour le chargement du modèle et, lorsque le
  fichier test existe, pour la reproduction des prédictions.

Lors de l'audit RC (Linux, Python 3.12, un worker de modèle), le scénario complet a
pris 19,27 secondes avec un budget nominal de 6 secondes par run et un pic RSS
d'environ 225 MiB. Les performances exactes dépendent de la machine.
