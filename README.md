# TP de Machine Learning - Mohammed NASSIRI

## Description
Ce projet implémente **9 algorithmes de Machine Learning** afin de prédire et classifier des coûts médicaux. Chaque algorithme est comparé entre une implémentation **From Scratch** et une implémentation via **Scikit-Learn**.

### Algorithmes de Régression
| # | Algorithme | Description |
|---|------------|-------------|
| 1 | **Régression Linéaire Simple** | Prédiction du coût à partir de l'âge |
| 2 | **Régression Linéaire Multiple** | Prédiction à partir de l'âge, du BMI et du nombre d'enfants |
| 3 | **Régression Polynomiale** | Modélisation non-linéaire (degré 2) entre l'âge et le coût |

### Algorithmes de Classification
| # | Algorithme | Description |
|---|------------|-------------|
| 4 | **Régression Logistique** | Classification binaire via la fonction sigmoïde |
| 5 | **KNN (K Plus Proches Voisins)** | Vote majoritaire des K=5 voisins les plus proches |
| 6 | **SVM (Machine à Vecteurs de Support)** | Hyperplan optimal, comparaison noyau linéaire vs RBF |

### Arbres de Décision
| # | Algorithme | Description |
|---|------------|-------------|
| 7 | **ID3** | Arbre de décision basé sur le Gain d'Information (Entropie) |
| 8 | **CART** | Arbre de décision basé sur l'Impureté de Gini |
| 9 | **C4.5** | Arbre de décision basé sur le Ratio de Gain |

## Structure du Projet

```text
.
├── data/
│   └── medical_costs.csv              # Dataset (500 échantillons)
├── graphs/                            # Graphes générés
│   ├── regression_lineaire_simple.png
│   ├── regression_multiple.png
│   ├── regression_polynomiale.png
│   ├── regression_logistique.png
│   ├── knn.png
│   ├── svm.png
│   ├── id3.png
│   ├── cart.png
│   ├── c45.png
│   └── graphe_combine_final.png       # Vue d'ensemble 3×3
├── testing/
│   └── main.py                        # Script de test automatisé
├── regression_mnassiri.ipynb           # Notebook : Régression
├── classification_mnassiri.ipynb       # Notebook : Classification
├── arbres_decision_mnassiri.ipynb      # Notebook : Arbres de Décision
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Utilisation

### Prérequis
```bash
poetry install
# OU
pip install -r requirements.txt
```

### Option A : Test Automatisé en Terminal
```bash
python testing/main.py
```

### Option B : Notebooks Jupyter
Ouvrez le notebook correspondant à la catégorie souhaitée et lancez **Run All** :
- `regression_mnassiri.ipynb` — Algorithmes de régression
- `classification_mnassiri.ipynb` — Algorithmes de classification
- `arbres_decision_mnassiri.ipynb` — Arbres de décision
