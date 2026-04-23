#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import time
import os
import sys
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==========================================
# CONFIGURATION ET UTILITAIRES
# ==========================================
os.makedirs('data', exist_ok=True)
os.makedirs('graphs', exist_ok=True)

def log_step(message, delay=1.0):
    print(f"[*] {message}...", end="", flush=True)
    time.sleep(delay)
    print(" TERMINÉ")

# ==========================================
# IMPLÉMENTATIONS FROM SCRATCH
# ==========================================

class RegressionFromScratch:
    def __init__(self):
        self.theta = None

    def fit(self, X_train, y_train):
        X_b = np.c_[np.ones((len(X_train), 1)), X_train]
        self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

    def predict(self, X_test):
        X_b = np.c_[np.ones((len(X_test), 1)), X_test]
        return X_b.dot(self.theta)


class LogisticRegressionFromScratch:
    """
    Régression logistique implémentée manuellement via la descente de gradient.
    Utilise la fonction sigmoïde pour modéliser la probabilité d'appartenance à une classe.
    """
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        # Clip pour éviter les dépassements numériques (overflow)
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            # Prédiction linéaire puis application de la sigmoïde
            z = X_train.dot(self.weights) + self.bias
            predictions = self._sigmoid(z)

            # Calcul des gradients
            dw = (1 / n_samples) * X_train.T.dot(predictions - y_train)
            db = (1 / n_samples) * np.sum(predictions - y_train)

            # Mise à jour des poids
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X_test):
        z = X_test.dot(self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X_test):
        return (self.predict_proba(X_test) >= 0.5).astype(int)


class KNNFromScratch:
    """
    K Plus Proches Voisins (KNN) implémenté manuellement.
    Utilise la distance euclidienne et le vote majoritaire pour la classification.
    """
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        X_test = np.array(X_test)
        predictions = []
        for x in X_test:
            # Calcul de la distance euclidienne vers tous les points d'entraînement
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            # Sélection des K voisins les plus proches
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            # Vote majoritaire
            prediction = int(np.round(np.mean(k_labels)))
            predictions.append(prediction)
        return np.array(predictions)


try:
    print("\n" + "==="*15)
    print("   LANCEMENT DES TESTS DE MACHINE LEARNING")
    print("==="*15 + "\n")
    
    log_step("Chargement du dataset 'medical_costs.csv'", 0.5)
    
    try:
        df = pd.read_csv('data/medical_costs.csv')
    except FileNotFoundError:
        print("\n[ERREUR] Le fichier 'medical_costs.csv' est introuvable dans le dossier 'data'. Annulation.")
        sys.exit(1)

    # === Préparation des données (Régression) ===
    X_simple = df[['age']].values
    y = df[['cost']].values
    X_multi = df[['age', 'bmi', 'children']].values

    # === Préparation des données (Classification) ===
    # Création d'une variable cible binaire : coût élevé (1) ou faible (0)
    # Le seuil est la médiane du coût médical
    median_cost = np.median(df['cost'].values)
    y_class = (df['cost'].values >= median_cost).astype(int)
    X_class = df[['age', 'bmi']].values  # 2 variables pour la visualisation 2D

    # Séparation en ensembles d'entraînement et de test (80/20)
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42
    )

    # Normalisation des caractéristiques (essentiel pour KNN et SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_cls)
    X_test_scaled = scaler.transform(X_test_cls)

    # Préparation de la grille pour les frontières de décision (classification)
    h = 0.5  # Pas de la grille
    x_min, x_max = X_class[:, 0].min() - 2, X_class[:, 0].max() + 2
    y_min, y_max = X_class[:, 1].min() - 2, X_class[:, 1].max() + 2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_scaled = scaler.transform(grid_points)

    # Couleurs pour les classes (pour toutes les visualisations de classification)
    class_colors = np.where(y_class == 1, '#e74c3c', '#3498db')

    log_step("Préparation des données de classification", 0.5)

    # === Figure combinée (2 lignes × 3 colonnes) ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.canvas.manager.set_window_title('Aperçu des Modèles de Machine Learning')

    # ==========================================
    # 1. RÉGRESSION LINÉAIRE SIMPLE
    # ==========================================
    print("\n" + "-"*45)
    print(" 1 : RÉGRESSION LINÉAIRE SIMPLE")
    print("-"*45 + '\n')
    log_step("Calcul des modèles (From Scratch vs Bibliothèque)")
    
    model_simple = RegressionFromScratch()
    model_simple.fit(X_simple, y)
    
    sk_simple = LinearRegression().fit(X_simple, y)
    
    X_plot = np.linspace(X_simple.min(), X_simple.max(), 100).reshape(-1, 1)
    axes[0, 0].scatter(X_simple, y, color='gray', alpha=0.5, label='Données Réelles')
    axes[0, 0].plot(X_plot, model_simple.predict(X_plot), color='blue', linewidth=4, label='From Scratch')
    axes[0, 0].plot(X_plot, sk_simple.predict(X_plot), color='red', linestyle='dashed', linewidth=2, label='Bibliothèque')
    axes[0, 0].set_title('Régression Linéaire Simple (Âge vs Coût)')
    axes[0, 0].set_xlabel('Âge')
    axes[0, 0].set_ylabel('Coût Médical')
    axes[0, 0].legend()
    
    plt.figure()
    plt.scatter(X_simple, y, color='gray', alpha=0.5)
    plt.plot(X_plot, model_simple.predict(X_plot), color='blue', linewidth=2)
    plt.title('Régression Linéaire Simple')
    plt.xlabel('Âge')
    plt.ylabel('Coût Médical')
    plt.savefig('graphs/regression_lineaire_simple.png')
    plt.close()
    log_step("Sauvegarde du graphique dans 'graphs/'", 0.5)

    # ==========================================
    # 2. RÉGRESSION LINÉAIRE MULTIPLE
    # ==========================================
    print("\n" + "-"*45)
    print(" 2 : RÉGRESSION LINÉAIRE MULTIPLE")
    print("-"*45 + '\n')
    log_step("Calcul des modèles (From Scratch vs Bibliothèque)")
    
    model_multi = RegressionFromScratch()
    model_multi.fit(X_multi, y)
    
    sk_multi = LinearRegression().fit(X_multi, y)
    
    y_pred_scratch = model_multi.predict(X_multi)
    axes[0, 1].scatter(y, y_pred_scratch, color='green', alpha=0.5, label='Prédictions (From Scratch)')
    axes[0, 1].plot([y.min(), y.max()], [y.min(), y.max()], color='black', linestyle='dashed', lw=2, label='Ajustement Parfait')
    axes[0, 1].set_title('Régression Multiple (Réel vs Prédit)')
    axes[0, 1].set_xlabel('Coût Réel')
    axes[0, 1].set_ylabel('Coût Prédit')
    axes[0, 1].legend()

    plt.figure()
    plt.scatter(y, y_pred_scratch, color='green', alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='black', linestyle='dashed', lw=2)
    plt.title('Régression Multiple (Réel vs Prédit)')
    plt.xlabel('Coût Réel')
    plt.ylabel('Coût Prédit')
    plt.savefig('graphs/regression_multiple.png')
    plt.close()
    log_step("Sauvegarde du graphique dans 'graphs/'", 0.5)

    # ==========================================
    # 3. RÉGRESSION POLYNOMIALE
    # ==========================================
    print("\n" + "-"*45)
    print(" 3 : RÉGRESSION POLYNOMIALE")
    print("-"*45 + '\n')
    log_step("Calcul des modèles (From Scratch vs Bibliothèque)")
    
    X_poly_scratch = np.c_[X_simple, X_simple**2]
    model_poly = RegressionFromScratch()
    model_poly.fit(X_poly_scratch, y)
    
    poly_feats = PolynomialFeatures(degree=2, include_bias=False)
    X_poly_sk = poly_feats.fit_transform(X_simple)
    sk_poly = LinearRegression().fit(X_poly_sk, y)
    
    X_plot_poly = np.c_[X_plot, X_plot**2]
    X_plot_poly_sk = poly_feats.transform(X_plot)
    
    axes[0, 2].scatter(X_simple, y, color='gray', alpha=0.5, label='Données Réelles')
    axes[0, 2].plot(X_plot, model_poly.predict(X_plot_poly), color='purple', linewidth=4, label='From Scratch')
    axes[0, 2].plot(X_plot, sk_poly.predict(X_plot_poly_sk), color='orange', linestyle='dashed', linewidth=2, label='Bibliothèque')
    axes[0, 2].set_title('Régression Polynomiale (Degré 2)')
    axes[0, 2].set_xlabel('Âge')
    axes[0, 2].set_ylabel('Coût Médical')
    axes[0, 2].legend()

    plt.figure()
    plt.scatter(X_simple, y, color='gray', alpha=0.5)
    plt.plot(X_plot, model_poly.predict(X_plot_poly), color='purple', linewidth=2)
    plt.title('Régression Polynomiale')
    plt.xlabel('Âge')
    plt.ylabel('Coût Médical')
    plt.savefig('graphs/regression_polynomiale.png')
    plt.close()
    log_step("Sauvegarde du graphique dans 'graphs/'", 0.5)

    # ==========================================
    # 4. RÉGRESSION LOGISTIQUE (Classification)
    # ==========================================
    print("\n" + "-"*45)
    print(" 4 : RÉGRESSION LOGISTIQUE")
    print("-"*45 + '\n')
    log_step("Calcul des modèles (From Scratch vs Bibliothèque)")

    # Modèle From Scratch (descente de gradient)
    model_log_scratch = LogisticRegressionFromScratch(learning_rate=0.1, n_iterations=1000)
    model_log_scratch.fit(X_train_scaled, y_train_cls)
    y_pred_log_scratch = model_log_scratch.predict(X_test_scaled)
    acc_log_scratch = accuracy_score(y_test_cls, y_pred_log_scratch)

    # Modèle Bibliothèque (Sklearn)
    model_log_sklearn = SklearnLogisticRegression(max_iter=1000)
    model_log_sklearn.fit(X_train_scaled, y_train_cls)
    y_pred_log_sklearn = model_log_sklearn.predict(X_test_scaled)
    acc_log_sklearn = accuracy_score(y_test_cls, y_pred_log_sklearn)

    print(f"  [RÉSULTAT] Précision From Scratch : {acc_log_scratch:.1%}")
    print(f"  [RÉSULTAT] Précision Bibliothèque : {acc_log_sklearn:.1%}")

    # Frontières de décision
    Z_log_scratch = model_log_scratch.predict(grid_points_scaled).reshape(xx.shape)
    Z_log_sklearn = model_log_sklearn.predict(grid_points_scaled).reshape(xx.shape)

    # -- Graphique combiné (sous-graphe) --
    axes[1, 0].contourf(xx, yy, Z_log_scratch, alpha=0.3, cmap='coolwarm',
                         levels=np.linspace(-0.5, 1.5, 3))
    axes[1, 0].contour(xx, yy, Z_log_sklearn, colors='red', linestyles='dashed',
                        linewidths=2.5, levels=[0.5])
    axes[1, 0].scatter(X_class[:, 0], X_class[:, 1], c=class_colors,
                        edgecolors='black', s=20, alpha=0.6, linewidths=0.5)
    axes[1, 0].set_title('Régression Logistique')
    axes[1, 0].set_xlabel('Âge')
    axes[1, 0].set_ylabel('BMI')
    axes[1, 0].legend(handles=[
        Patch(facecolor='#d35400', alpha=0.3, label=f'From Scratch ({acc_log_scratch:.1%})'),
        Line2D([0], [0], color='red', linestyle='dashed', linewidth=2,
               label=f'Bibliothèque ({acc_log_sklearn:.1%})'),
        Patch(facecolor='#e74c3c', label='Coût Élevé'),
        Patch(facecolor='#3498db', label='Coût Faible'),
    ], fontsize=7, loc='upper left')

    # -- Graphique individuel --
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z_log_scratch, alpha=0.3, cmap='coolwarm',
                 levels=np.linspace(-0.5, 1.5, 3))
    plt.scatter(X_class[:, 0], X_class[:, 1], c=class_colors,
                edgecolors='black', s=20, alpha=0.6, linewidths=0.5)
    plt.title('Régression Logistique (From Scratch)')
    plt.xlabel('Âge')
    plt.ylabel('BMI')
    plt.legend(handles=[
        Patch(facecolor='#e74c3c', label='Coût Élevé'),
        Patch(facecolor='#3498db', label='Coût Faible'),
    ])
    plt.savefig('graphs/regression_logistique.png')
    plt.close()
    log_step("Sauvegarde du graphique dans 'graphs/'", 0.5)

    # ==========================================
    # 5. KNN (K Plus Proches Voisins)
    # ==========================================
    print("\n" + "-"*45)
    print(" 5 : KNN (K PLUS PROCHES VOISINS)")
    print("-"*45 + '\n')
    log_step("Calcul des modèles (From Scratch vs Bibliothèque)")

    # Modèle From Scratch
    model_knn_scratch = KNNFromScratch(k=5)
    model_knn_scratch.fit(X_train_scaled, y_train_cls)
    y_pred_knn_scratch = model_knn_scratch.predict(X_test_scaled)
    acc_knn_scratch = accuracy_score(y_test_cls, y_pred_knn_scratch)

    # Modèle Bibliothèque (Sklearn)
    model_knn_sklearn = KNeighborsClassifier(n_neighbors=5)
    model_knn_sklearn.fit(X_train_scaled, y_train_cls)
    y_pred_knn_sklearn = model_knn_sklearn.predict(X_test_scaled)
    acc_knn_sklearn = accuracy_score(y_test_cls, y_pred_knn_sklearn)

    print(f"  [RÉSULTAT] Précision From Scratch : {acc_knn_scratch:.1%}")
    print(f"  [RÉSULTAT] Précision Bibliothèque : {acc_knn_sklearn:.1%}")

    # Frontières de décision (le KNN From Scratch peut prendre quelques secondes)
    print("  [*] Calcul des frontières de décision KNN...", end="", flush=True)
    Z_knn_scratch = model_knn_scratch.predict(grid_points_scaled).reshape(xx.shape)
    Z_knn_sklearn = model_knn_sklearn.predict(grid_points_scaled).reshape(xx.shape)
    print(" TERMINÉ")

    # -- Graphique combiné (sous-graphe) --
    axes[1, 1].contourf(xx, yy, Z_knn_scratch, alpha=0.3, cmap='coolwarm',
                         levels=np.linspace(-0.5, 1.5, 3))
    axes[1, 1].contour(xx, yy, Z_knn_sklearn, colors='red', linestyles='dashed',
                        linewidths=2.5, levels=[0.5])
    axes[1, 1].scatter(X_class[:, 0], X_class[:, 1], c=class_colors,
                        edgecolors='black', s=20, alpha=0.6, linewidths=0.5)
    axes[1, 1].set_title('KNN (K=5)')
    axes[1, 1].set_xlabel('Âge')
    axes[1, 1].set_ylabel('BMI')
    axes[1, 1].legend(handles=[
        Patch(facecolor='#d35400', alpha=0.3, label=f'From Scratch ({acc_knn_scratch:.1%})'),
        Line2D([0], [0], color='red', linestyle='dashed', linewidth=2,
               label=f'Bibliothèque ({acc_knn_sklearn:.1%})'),
        Patch(facecolor='#e74c3c', label='Coût Élevé'),
        Patch(facecolor='#3498db', label='Coût Faible'),
    ], fontsize=7, loc='upper left')

    # -- Graphique individuel --
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z_knn_scratch, alpha=0.3, cmap='coolwarm',
                 levels=np.linspace(-0.5, 1.5, 3))
    plt.scatter(X_class[:, 0], X_class[:, 1], c=class_colors,
                edgecolors='black', s=20, alpha=0.6, linewidths=0.5)
    plt.title('KNN - K Plus Proches Voisins (K=5)')
    plt.xlabel('Âge')
    plt.ylabel('BMI')
    plt.legend(handles=[
        Patch(facecolor='#e74c3c', label='Coût Élevé'),
        Patch(facecolor='#3498db', label='Coût Faible'),
    ])
    plt.savefig('graphs/knn.png')
    plt.close()
    log_step("Sauvegarde du graphique dans 'graphs/'", 0.5)

    # ==========================================
    # 6. SVM (Machine à Vecteurs de Support)
    # ==========================================
    print("\n" + "-"*45)
    print(" 6 : SVM (MACHINE À VECTEURS DE SUPPORT)")
    print("-"*45 + '\n')
    log_step("Calcul des modèles (Noyau Linéaire vs Noyau RBF)")

    # Modèle avec Noyau Linéaire
    model_svm_linear = SVC(kernel='linear')
    model_svm_linear.fit(X_train_scaled, y_train_cls)
    y_pred_svm_linear = model_svm_linear.predict(X_test_scaled)
    acc_svm_linear = accuracy_score(y_test_cls, y_pred_svm_linear)

    # Modèle avec Noyau RBF (Gaussien)
    model_svm_rbf = SVC(kernel='rbf')
    model_svm_rbf.fit(X_train_scaled, y_train_cls)
    y_pred_svm_rbf = model_svm_rbf.predict(X_test_scaled)
    acc_svm_rbf = accuracy_score(y_test_cls, y_pred_svm_rbf)

    print(f"  [RÉSULTAT] Précision Noyau Linéaire : {acc_svm_linear:.1%}")
    print(f"  [RÉSULTAT] Précision Noyau RBF      : {acc_svm_rbf:.1%}")

    # Frontières de décision
    Z_svm_linear = model_svm_linear.predict(grid_points_scaled).reshape(xx.shape)
    Z_svm_rbf = model_svm_rbf.predict(grid_points_scaled).reshape(xx.shape)

    # -- Graphique combiné (sous-graphe) --
    axes[1, 2].contourf(xx, yy, Z_svm_linear, alpha=0.3, cmap='coolwarm',
                         levels=np.linspace(-0.5, 1.5, 3))
    axes[1, 2].contour(xx, yy, Z_svm_rbf, colors='red', linestyles='dashed',
                        linewidths=2.5, levels=[0.5])
    axes[1, 2].scatter(X_class[:, 0], X_class[:, 1], c=class_colors,
                        edgecolors='black', s=20, alpha=0.6, linewidths=0.5)
    axes[1, 2].set_title('SVM (Support Vector Machine)')
    axes[1, 2].set_xlabel('Âge')
    axes[1, 2].set_ylabel('BMI')
    axes[1, 2].legend(handles=[
        Patch(facecolor='#d35400', alpha=0.3, label=f'Noyau Linéaire ({acc_svm_linear:.1%})'),
        Line2D([0], [0], color='red', linestyle='dashed', linewidth=2,
               label=f'Noyau RBF ({acc_svm_rbf:.1%})'),
        Patch(facecolor='#e74c3c', label='Coût Élevé'),
        Patch(facecolor='#3498db', label='Coût Faible'),
    ], fontsize=7, loc='upper left')

    # -- Graphique individuel --
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z_svm_rbf, alpha=0.3, cmap='coolwarm',
                 levels=np.linspace(-0.5, 1.5, 3))
    plt.scatter(X_class[:, 0], X_class[:, 1], c=class_colors,
                edgecolors='black', s=20, alpha=0.6, linewidths=0.5)
    plt.title('SVM - Machine à Vecteurs de Support (Noyau RBF)')
    plt.xlabel('Âge')
    plt.ylabel('BMI')
    plt.legend(handles=[
        Patch(facecolor='#e74c3c', label='Coût Élevé'),
        Patch(facecolor='#3498db', label='Coût Faible'),
    ])
    plt.savefig('graphs/svm.png')
    plt.close()
    log_step("Sauvegarde du graphique dans 'graphs/'", 0.5)

    # ==========================================
    # 7. FINALISATION
    # ==========================================
    print("\n" + "==="*15)
    print("\n[SUCCÈS] Tous les modèles ont été générés.")
    print("[INFO] Fermez la fenêtre du graphe ou appuyez sur Ctrl+C.")
    
    fig.suptitle('MODELES DE MACHINE LEARNING', fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    print("\n[*] Sauvegarde du graphique combiné final à la racine...")
    fig.savefig('graphe_combine_final.png', dpi=150, bbox_inches='tight')
    print("[*] Fin du programme avec succès.")

except KeyboardInterrupt:
    print("\n\n [ATTENTION] Interruption clavier détectée (Ctrl+C).")
    print(" [*] Sauvegarde du graphique combiné final avant de quitter...")
    try:
        fig.savefig('graphe_combine_final.png', dpi=150, bbox_inches='tight')
        print(" [*] Sauvegarde réussie. Fermeture du programme.")
    except Exception as e:
        print(f" [ERREUR] Échec de la sauvegarde : {e}")
    sys.exit(0)