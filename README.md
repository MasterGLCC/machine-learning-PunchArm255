# TP de Régression - Mohammed NASSIRI

## Description
Ce projet implémente trois algorithmes de régression (Simple, Multiple, et Polynomiale) afin de prédire des coûts médicaux. L'objectif est de comparer une approche mathématique calculée directement via l'Équation Normale (From Scratch) avec les implémentations professionnelles de `scikit-learn`.

## Structure du Projet

```text
.
├── data/
│   └── medical_costs.csv        # Dataset d'entraînement
├── graphs/                      # Dossier de sortie pour les graphes générés
├── regression_mnassiri.ipynb    # Notebook interactif (Livrable final)
├── main.py                      # Script dynamique pour exécution et tests manuels
├── pyproject.toml               # Configuration des dépendances (Poetry)
├── requirements.txt             # Configuration des dépendances (pip)
└── README.md                    # Documentation du projet
```

## Déploiement et Tests Manuels

### Prérequis
Ce projet utilise Poetry ou pip pour une gestion propre des dépendances. Initialisez l'environnement avec ces commandes :

```bash
poetry install
# OU
pip install -r requirements.txxt
```

### Option A : Test Manuel en Terminal
Pour tester la logique pure, visualiser les logs d'exécution et générer physiquement les fichiers PNG dans le dossier `graphs/`, utilisez le script principal :

```bash
python3 main.py
```

### Option B : Évaluation via Jupyter Notebook
Le fichier interactif contient tout le code segmenté, commenté et les rendus graphiques intégrés.
1. Ouvrez `regression_mnassiri.ipynb` dans VSCode ou votre environnement Jupyter.
2. Lancez l'exécution globale (Run All).