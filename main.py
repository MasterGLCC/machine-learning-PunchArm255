#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ==========================================
# CONFIGURATION ET UTILITAIRES
# ==========================================
os.makedirs('data', exist_ok=True)
os.makedirs('graphs', exist_ok=True)

def log_step(message, delay=1.0):
    print(f"[*] {message}...", end="", flush=True)
    time.sleep(delay)
    print(" TERMINÉ")

class RegressionFromScratch:
    def __init__(self):
        self.theta = None

    def fit(self, X_train, y_train):
        X_b = np.c_[np.ones((len(X_train), 1)), X_train]
        self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

    def predict(self, X_test):
        X_b = np.c_[np.ones((len(X_test), 1)), X_test]
        return X_b.dot(self.theta)

try:
    print("\n" + "==="*15)
    print("   LANCEMENT DU TEST DE RÉGRESSION")
    print("==="*15 + "\n")
    
    log_step("Chargement du dataset 'medical_costs.csv'", 0.5)
    
    try:
        df = pd.read_csv('data/medical_costs.csv')
    except FileNotFoundError:
        print("\n[ERREUR] Le fichier 'medical_costs.csv' est introuvable dans le dossier 'data'. Annulation.")
        sys.exit(1)

    X_simple = df[['age']].values
    y = df[['cost']].values
    X_multi = df[['age', 'bmi', 'children']].values

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.canvas.manager.set_window_title('Aperçu des Modèles de Régression')

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
    axes[0].scatter(X_simple, y, color='gray', alpha=0.5, label='Données Réelles')
    axes[0].plot(X_plot, model_simple.predict(X_plot), color='blue', linewidth=4, label='From Scratch')
    axes[0].plot(X_plot, sk_simple.predict(X_plot), color='red', linestyle='dashed', linewidth=2, label='Bibliothèque')
    axes[0].set_title('Régression Linéaire Simple (Âge vs Coût)')
    axes[0].set_xlabel('Âge')
    axes[0].set_ylabel('Coût Médical')
    axes[0].legend()
    
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
    axes[1].scatter(y, y_pred_scratch, color='green', alpha=0.5, label='Prédictions (From Scratch)')
    axes[1].plot([y.min(), y.max()], [y.min(), y.max()], color='black', linestyle='dashed', lw=2, label='Ajustement Parfait')
    axes[1].set_title('Régression Multiple (Réel vs Prédit)')
    axes[1].set_xlabel('Coût Réel')
    axes[1].set_ylabel('Coût Prédit')
    axes[1].legend()

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
    
    axes[2].scatter(X_simple, y, color='gray', alpha=0.5, label='Données Réelles')
    axes[2].plot(X_plot, model_poly.predict(X_plot_poly), color='purple', linewidth=4, label='From Scratch')
    axes[2].plot(X_plot, sk_poly.predict(X_plot_poly_sk), color='orange', linestyle='dashed', linewidth=2, label='Bibliothèque')
    axes[2].set_title('Régression Polynomiale (Degré 2)')
    axes[2].set_xlabel('Âge')
    axes[2].set_ylabel('Coût Médical')
    axes[2].legend()

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
    # 4. FINALISATION
    # ==========================================
    print("\n" + "==="*15)
    print("\n[SUCCÈS] Tous les modèles ont été générés.")
    print("[INFO] Fermez la fenêtre du graphe ou appuyez sur Ctrl+C.")
    
    plt.tight_layout()
    plt.show()

    print("\n[*] Sauvegarde du graphique combiné final à la racine...")
    fig.savefig('graphe_combine_final.png')
    print("[*] Fin du programme avec succès.")

except KeyboardInterrupt:
    print("\n\n [ATTENTION] Interruption clavier détectée (Ctrl+C).")
    print(" [*] Sauvegarde du graphique combiné final avant de quitter...")
    try:
        fig.savefig('graphe_combine_final.png')
        print(" [*] Sauvegarde réussie. Fermeture du programme.")
    except Exception as e:
        print(f" [ERREUR] Échec de la sauvegarde : {e}")
    sys.exit(0)