# TP de Machine Learning - Mohammed NASSIRI

## Description
Ce projet implémente **6 algorithmes de Machine Learning** afin de prédire et classifier des coûts médicaux. L'objectif est de comparer des approches mathématiques calculées directement (From Scratch) avec les implémentations professionnelles de `scikit-learn`.

### Algorithmes de Régression
| # | Algorithme | Approche | Description |
|---|------------|----------|-------------|
| 1 | **Régression Linéaire Simple** | From Scratch + Sklearn | Prédiction du coût uniquement à partir de l'âge |
| 2 | **Régression Linéaire Multiple** | From Scratch + Sklearn | Prédiction du coût à partir de l'âge, du BMI et du nombre d'enfants |
| 3 | **Régression Polynomiale** | From Scratch + Sklearn | Modélisation d'une relation non-linéaire (degré 2) entre l'âge et le coût |

### Algorithmes de Classification
Pour la classification, une variable cible binaire est créée : **coût élevé** (≥ médiane) ou **coût faible** (< médiane).

| # | Algorithme | Approche | Description |
|---|------------|----------|-------------|
| 4 | **Régression Logistique** | From Scratch + Sklearn | Classification binaire via la fonction sigmoïde et la descente de gradient |
| 5 | **KNN (K Plus Proches Voisins)** | From Scratch + Sklearn | Classification par vote majoritaire des K=5 voisins les plus proches |
| 6 | **SVM (Machine à Vecteurs de Support)** | Sklearn (Linéaire vs RBF) | Classification par hyperplan optimal, comparaison de deux noyaux |

## Structure du Projet

```text
.
├── data/
│   └── medical_costs.csv        # Dataset d'entraînement (500 échantillons)
├── graphs/                      # Dossier de sortie pour les graphes générés
│   ├── regression_lineaire_simple.png
│   ├── regression_multiple.png
│   ├── regression_polynomiale.png
│   ├── regression_logistique.png
│   ├── knn.png
│   └── svm.png
├── graphe_combine_final.png     # Vue d'ensemble des 6 modèles (2×3)
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
pip install -r requirements.txt
```

### Option A : Test Manuel en Terminal
Pour tester la logique pure, visualiser les logs d'exécution et générer physiquement les fichiers PNG dans le dossier `graphs/`, utilisez le script principal :

```bash
python3 main.py
```

Ce script exécute les 6 algorithmes séquentiellement, affiche les résultats de précision dans le terminal, et génère un graphique combiné final.

### Option B : Évaluation via Jupyter Notebook
Le fichier interactif contient tout le code segmenté, commenté et les rendus graphiques intégrés.
1. Ouvrez `regression_mnassiri.ipynb` dans VSCode ou votre environnement Jupyter.
2. Lancez l'exécution globale (Run All).


TODO: ID3 - C4.5 - C5.0 - CART
