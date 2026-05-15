# TP de Machine Learning - Mohammed NASSIRI

## Description
Ce projet implémente **14 algorithmes de Machine Learning** afin de prédire et classifier des coûts médicaux. Chaque algorithme est comparé entre une implémentation **From Scratch** et une implémentation via **Scikit-Learn**.

### Algorithmes de Régression
| # | Algorithme | Description |
|---|------------|-------------|
| 1 | **Régression Linéaire Simple** | Prédiction du coût à partir de l'âge |
| 2 | **Régression Linéaire Multiple** | Prédiction à partir de l'âge, du BMI et du nombre d'enfants |
| 3 | **Régression Polynomiale** | Modélisation non-linéaire (degré 2) |

### Algorithmes de Classification
| # | Algorithme | Description |
|---|------------|-------------|
| 4 | **Régression Logistique** | Classification binaire via la fonction sigmoïde |
| 5 | **KNN** | Vote majoritaire des K=5 voisins les plus proches |
| 6 | **SVM** | Hyperplan optimal (noyau linéaire vs RBF) |

### Arbres de Décision
| # | Algorithme | Description |
|---|------------|-------------|
| 7 | **ID3** | Gain d'Information (Entropie) |
| 8 | **CART** | Impureté de Gini |
| 9 | **C4.5** | Ratio de Gain |

### Apprentissage Non Supervisé
| # | Algorithme | Description |
|---|------------|-------------|
| 10 | **DBSCAN** | Clustering par densité |
| 12 | **PCA** | Réduction de dimensionnalité (Composantes Principales) |

### Classification Avancée
| # | Algorithme | Description |
|---|------------|-------------|
| 11 | **Naive Bayes** | Classification bayésienne gaussienne |
| 13 | **XGBoost** | Gradient Boosting avec souches de décision |

### Apprentissage par Renforcement
| # | Algorithme | Description |
|---|------------|-------------|
| 14 | **Q-Learning** | Agent apprenant sur un GridWorld 5×5 |

## Structure du Projet

```text
.
├── data/
│   └── medical_costs.csv
├── graphs/                              # Graphes générés (14 individuels + 1 combiné)
├── testing/
│   └── main.py                          # Script de test automatisé (14 algorithmes)
├── regression_mnassiri.ipynb            # Régression
├── classification_mnassiri.ipynb        # Classification (LogReg, KNN, SVM)
├── arbres_decision_mnassiri.ipynb       # Arbres de Décision (ID3, CART, C4.5)
├── non_supervise_mnassiri.ipynb         # DBSCAN + PCA
├── bayes_boosting_mnassiri.ipynb        # Naive Bayes + XGBoost
├── renforcement_mnassiri.ipynb          # Q-Learning
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Utilisation

### Prérequis
```bash
pip install -r requirements.txt
```

### Option A : Test Automatisé
```bash
python testing/main.py
```

### Option B : Notebooks Jupyter
Ouvrez le notebook correspondant et lancez **Run All**.
