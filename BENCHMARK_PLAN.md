# Plan de benchmark

## 1. Question principale

Le contrôleur adaptatif trouve-t-il de meilleurs pipelines plus rapidement qu’une recherche standard sous le même budget ?

## 2. Variantes comparées

### A — Random Search

- espace identique ;
- budget identique ;
- aucune mémoire ;
- aucune allocation adaptative.

### B — Optuna simple

- une étude globale ;
- espace conditionnel ;
- pruning standard ;
- aucune allocation par famille.

### C — Adaptive Scheduler

- optimizers par famille ;
- multi-fidélité ;
- allocation adaptative ;
- aucune mémoire.

### D — Adaptive Scheduler + Memory

- mêmes composants que C ;
- warm start par datasets similaires.

## 3. Datasets

Commencer par une suite interne synthétique :

- linéaire ;
- non linéaire ;
- catégoriel ;
- haute cardinalité ;
- valeurs manquantes MCAR ;
- valeurs manquantes structurées ;
- déséquilibre ;
- petit n, grand p ;
- grand n, petit p.

Ajouter ensuite des datasets publics tabulaires clairement licenciés.

## 4. Budgets

Pour chaque dataset :

- 1 minute ;
- 5 minutes ;
- 15 minutes ;
- 30 minutes.

Les budgets doivent être appliqués de manière comparable.

## 5. Seeds

Au moins cinq seeds par combinaison pour les résultats sérieux.

## 6. Mesures

- meilleur score à chaque instant ;
- temps pour atteindre 95 % du meilleur score observé ;
- score final ;
- nombre d’essais ;
- nombre d’essais échoués ;
- coût CPU ;
- mémoire ;
- diversité des finalistes ;
- stabilité entre seeds ;
- gain du warm start.

## 7. Anytime Performance

Mesure principale :

```text
aire sous la courbe :
temps → meilleur score obtenu
```

La baseline de référence doit être soustraite lorsque pertinent.

## 8. Équité

Toutes les variantes doivent utiliser :

- mêmes données ;
- mêmes splits ;
- même espace de pipelines ;
- mêmes ressources ;
- même budget ;
- mêmes métriques ;
- mêmes contraintes de modèles disponibles.

## 9. Validation du méta-apprentissage

La mémoire ne doit pas voir les résultats du dataset évalué.

Utiliser une séparation par dataset :

```text
train meta-memory : datasets 1..N-1
test : dataset N
```

## 10. Rapport attendu

Le rapport doit contenir :

- tableaux agrégés ;
- courbes anytime ;
- distributions par seed ;
- coût ;
- taux d’échec ;
- ablations ;
- limites ;
- datasets où l’approche échoue.

## 11. Ablations

Comparer :

- sans multi-fidélité ;
- sans pénalité coût ;
- sans diversité ;
- sans mémoire ;
- sans landmarks ;
- UCB contre allocation uniforme.

## 12. Critère de succès initial

Le scheduler adaptatif est considéré prometteur s’il :

- atteint plus vite un score proche du meilleur ;
- évite une fraction importante d’essais coûteux ;
- ne dégrade pas significativement le score final moyen ;
- reste robuste sur plusieurs types de datasets.
