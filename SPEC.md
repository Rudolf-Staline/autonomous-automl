# Spécification produit

## 1. Vision

Construire un moteur AutoML autonome pour données tabulaires qui cherche le meilleur pipeline complet sous contrainte de budget.

Le pipeline complet comprend notamment :

- préparation du schéma ;
- traitement des valeurs manquantes ;
- encodage ;
- transformations numériques ;
- sélection éventuelle de variables ;
- modèle ;
- hyperparamètres ;
- calibration éventuelle ;
- stratégie d’ensemble éventuelle.

Le système doit privilégier une recherche intelligente plutôt qu’un produit cartésien exhaustif.

## 2. Utilisateur cible

- data scientist ;
- étudiant ou chercheur ;
- participant à une compétition tabulaire ;
- ingénieur souhaitant obtenir une baseline robuste ;
- utilisateur disposant de CSV mais sans pipeline établi.

## 3. Entrées

### 3.1 Obligatoires

- chemin vers un CSV d’entraînement ;
- colonne cible ;
- budget de calcul.

### 3.2 Facultatives

- CSV de test ;
- tâche :
  - `auto`
  - `binary_classification`
  - `multiclass_classification`
  - `regression`
- métrique :
  - `auto`
  - `roc_auc`
  - `average_precision`
  - `log_loss`
  - `accuracy`
  - `balanced_accuracy`
  - `f1`
  - `macro_f1`
  - `rmse`
  - `mae`
  - `r2`
- colonne identifiant ;
- colonne groupe ;
- colonne temporelle ;
- split ou fold prédéfini ;
- limite CPU ;
- limite mémoire indicative ;
- activation GPU ;
- seed ;
- stratégie d’optimisation :
  - `accuracy`
  - `balanced`
  - `fast`
- répertoire de sortie.

## 4. Sorties

Chaque run doit produire :

- manifeste complet ;
- profil du dataset ;
- plan de validation ;
- historique de tous les essais ;
- leaderboard ;
- meilleur `PipelineSpec` ;
- meilleur pipeline sérialisé ;
- prédictions out-of-fold lorsque possible ;
- prédictions du fichier test lorsque fourni ;
- métriques ;
- mesures de temps ;
- erreurs des essais ;
- rapport HTML ;
- notebook final ;
- informations d’environnement ;
- logs.

## 5. Tâches supportées en V1

- classification binaire ;
- classification multiclasse ;
- régression.

## 6. Types de colonnes supportés

- numériques entières ;
- numériques flottantes ;
- booléennes ;
- catégorielles ;
- chaînes à cardinalité faible ou moyenne ;
- dates simples convertibles ;
- colonnes entièrement manquantes ;
- colonnes constantes.

Le texte libre long doit être détecté et exclu ou signalé dans la V1.

## 7. Profilage

Le profiler doit calculer au minimum :

- nombre de lignes ;
- nombre de colonnes ;
- types inférés ;
- ratio de valeurs manquantes par colonne ;
- cardinalité ;
- ratio d’unicité ;
- colonnes constantes ;
- colonnes presque constantes ;
- colonnes candidates identifiant ;
- colonnes candidates temps ;
- colonnes candidates groupe ;
- distribution de la cible ;
- déséquilibre ;
- skewness numérique ;
- valeurs infinies ;
- doublons ;
- doublons potentiellement traversant les folds ;
- colonnes suspectes de fuite ;
- taille mémoire ;
- landmarks rapides.

## 8. Détection des risques de fuite

La V1 doit détecter ou signaler :

- colonne identique à la cible ;
- colonne presque identique à la cible ;
- colonne dont le nom suggère la cible ou le résultat ;
- identifiant unique ;
- doublons exacts ;
- colonnes calculées après l’événement prédit ;
- groupe présent dans plusieurs folds lorsque `group_column` est fourni ;
- dates postérieures à l’instant de prédiction lorsque la sémantique est fournie ;
- target encoding non cross-fitté ;
- imputation ajustée avant le split.

La détection heuristique ne doit pas supprimer silencieusement des colonnes douteuses. Elle doit enregistrer la décision et son niveau de confiance.

## 9. Validation

Le système doit sélectionner automatiquement entre :

- `StratifiedKFold` ;
- `KFold` ;
- `GroupKFold` ;
- `StratifiedGroupKFold` si disponible et approprié ;
- `TimeSeriesSplit` pour un mode temporel explicite ;
- split prédéfini.

Le choix doit être enregistré dans `ValidationPlan`.

## 10. Métriques

Le système doit :

- choisir une métrique par défaut selon la tâche ;
- maximiser une représentation interne cohérente ;
- conserver la métrique d’origine ;
- conserver les métriques secondaires ;
- gérer les prédictions de probabilités lorsque nécessaires ;
- refuser proprement une métrique incompatible.

Par défaut :

- classification binaire équilibrée : ROC AUC ;
- classification binaire très déséquilibrée : Average Precision ;
- classification multiclasse : macro F1 ou log loss selon configuration ;
- régression : RMSE.

## 11. Composants de prétraitement V1

### Numérique

- aucune imputation si le modèle supporte les valeurs manquantes ;
- moyenne ;
- médiane ;
- constante ;
- médiane + indicateur de valeur manquante ;
- KNN sur petits datasets ;
- imputation itérative optionnelle ;
- standardisation ;
- robust scaling ;
- aucune mise à l’échelle pour les arbres.

### Catégoriel

- catégorie constante `__MISSING__` ;
- mode ;
- one-hot avec gestion des catégories inconnues ;
- ordinal avec valeur inconnue ;
- fréquence ;
- encodage natif CatBoost lorsque disponible ;
- target encoding cross-fitté, uniquement si correctement testé.

### Dates

- année ;
- mois ;
- jour ;
- jour de semaine ;
- durée relative à une date de référence explicite ;
- suppression de la colonne brute après extraction, sauf modèle compatible.

## 12. Modèles V1

Noyau obligatoire :

- DummyClassifier ;
- DummyRegressor ;
- LogisticRegression ;
- Ridge ;
- ElasticNet ;
- RandomForest ;
- ExtraTrees ;
- HistGradientBoosting.

Optionnels :

- XGBoost ;
- LightGBM ;
- CatBoost.

Chaque modèle doit être encapsulé dans un adapter qui expose :

- tâches supportées ;
- support des NaN ;
- support catégoriel natif ;
- support sparse ;
- coût estimé ;
- compatibilité ;
- espace d’hyperparamètres ;
- construction déterministe.

## 13. Recherche

Le système doit posséder quatre niveaux :

1. baselines rapides ;
2. exploration multi-familles ;
3. optimisation spécialisée ;
4. confirmation des finalistes.

Il doit utiliser :

- espaces conditionnels ;
- pruning ;
- multi-fidélité ;
- budget global ;
- budget par essai ;
- journalisation ;
- reprise.

## 14. Allocation adaptative

Chaque famille de pipeline possède un état :

- essais tentés ;
- essais réussis ;
- meilleur score ;
- amélioration récente ;
- coût moyen ;
- variance ;
- potentiel estimé ;
- priorité d’exploration.

Le contrôleur doit allouer le prochain essai en combinant :

- performance ;
- amélioration ;
- coût ;
- incertitude ;
- diversité ;
- exploration minimale.

Une politique simple de type UCB est acceptable pour la V1 si elle est réellement utilisée et testée.

## 15. Multi-fidélité

Niveaux minimaux :

- faible :
  - fraction réduite des données ;
  - 2 folds ;
  - peu d’itérations ;
- moyen :
  - fraction plus grande ;
  - 3 folds ;
- complet :
  - toutes les données ;
  - 5 folds ;
- confirmation :
  - toutes les données ;
  - plusieurs seeds pour les finalistes.

Les candidats sont promus selon le score ajusté par l’incertitude et la diversité.

## 16. Mémoire

La V1 doit enregistrer les métacaractéristiques et résultats.

Le warm start peut d’abord utiliser :

- distance normalisée entre métacaractéristiques ;
- meilleurs pipelines de datasets similaires ;
- ordre de priorité des familles.

Le méta-ranker avancé peut rester expérimental, mais la structure de stockage doit permettre son ajout.

## 17. Reprise

Un run interrompu doit pouvoir reprendre grâce à :

- base SQLite ;
- manifeste ;
- état du scheduler ;
- essais Optuna persistants ;
- artefacts intermédiaires atomiques.

Les essais complets ne doivent pas être recommencés.

## 18. Notebook

Le notebook doit être généré déterministiquement à partir des artefacts.

Il contient :

1. contexte du run ;
2. dépendances ;
3. chargement des données ;
4. profil du dataset ;
5. validation ;
6. meilleur pipeline ;
7. résultats ;
8. entraînement final ;
9. prédictions ;
10. sauvegarde.

Il doit être exécuté automatiquement et validé.

## 19. Non-objectifs V1

- battre tous les AutoML existants ;
- supporter tous les formats ;
- séries temporelles avancées ;
- NLP ;
- vision ;
- interface graphique ;
- infrastructure cloud ;
- fédération de données ;
- génération libre de code par LLM dans le chemin critique.
