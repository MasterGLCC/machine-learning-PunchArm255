#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import time, os, sys
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Naviguer vers la racine du projet
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.makedirs('data', exist_ok=True)
os.makedirs('graphs', exist_ok=True)

def log_step(msg, delay=1.0):
    print(f"[*] {msg}...", end="", flush=True)
    time.sleep(delay)
    print(" TERMINÉ")

# ==========================================
# IMPLÉMENTATIONS FROM SCRATCH
# ==========================================

class RegressionFromScratch:
    def __init__(self):
        self.theta = None
    def fit(self, X, y):
        X_b = np.c_[np.ones((len(X), 1)), X]
        self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    def predict(self, X):
        X_b = np.c_[np.ones((len(X), 1)), X]
        return X_b.dot(self.theta)

class LogisticRegressionFromScratch:
    def __init__(self, lr=0.1, n_iter=1000):
        self.lr = lr; self.n_iter = n_iter; self.w = None; self.b = None
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    def fit(self, X, y):
        n, m = X.shape; self.w = np.zeros(m); self.b = 0.0
        for _ in range(self.n_iter):
            p = self._sigmoid(X.dot(self.w) + self.b)
            self.w -= self.lr * (1/n) * X.T.dot(p - y)
            self.b -= self.lr * (1/n) * np.sum(p - y)
    def predict(self, X):
        return (self._sigmoid(X.dot(self.w) + self.b) >= 0.5).astype(int)

class KNNFromScratch:
    def __init__(self, k=5):
        self.k = k
    def fit(self, X, y):
        self.X = np.array(X); self.y = np.array(y)
    def predict(self, X):
        X = np.array(X); preds = []
        for x in X:
            d = np.sqrt(np.sum((self.X - x)**2, axis=1))
            idx = np.argsort(d)[:self.k]
            preds.append(int(np.round(np.mean(self.y[idx]))))
        return np.array(preds)

class ID3FromScratch:
    """Arbre de décision utilisant le Gain d'Information (Entropie)."""
    def __init__(self, max_depth=5):
        self.max_depth = max_depth; self.tree = None
    def _entropy(self, y):
        if len(y) == 0: return 0
        p = np.bincount(y, minlength=2) / len(y)
        return -np.sum([pi * np.log2(pi) for pi in p if pi > 0])
    def _info_gain(self, X, y, feat, thr):
        left = X[:, feat] <= thr; right = ~left
        if np.sum(left) == 0 or np.sum(right) == 0: return 0
        n = len(y)
        return self._entropy(y) - (np.sum(left)/n)*self._entropy(y[left]) - (np.sum(right)/n)*self._entropy(y[right])
    def _best_split(self, X, y):
        best_g, best_f, best_t = -1, None, None
        for f in range(X.shape[1]):
            for t in np.unique(X[:, f]):
                g = self._info_gain(X, y, f, t)
                if g > best_g: best_g, best_f, best_t = g, f, t
        return best_f, best_t
    def _build(self, X, y, depth=0):
        if len(np.unique(y)) == 1: return int(y[0])
        if depth >= self.max_depth or len(y) < 2: return int(np.round(np.mean(y)))
        f, t = self._best_split(X, y)
        if f is None: return int(np.round(np.mean(y)))
        left = X[:, f] <= t
        if np.sum(left) == 0 or np.sum(~left) == 0: return int(np.round(np.mean(y)))
        return {'f': f, 't': t, 'l': self._build(X[left], y[left], depth+1), 'r': self._build(X[~left], y[~left], depth+1)}
    def fit(self, X, y): self.tree = self._build(X, y)
    def _pred1(self, x, node):
        if not isinstance(node, dict): return node
        return self._pred1(x, node['l'] if x[node['f']] <= node['t'] else node['r'])
    def predict(self, X): return np.array([self._pred1(x, self.tree) for x in X])

class CARTFromScratch:
    """Arbre de décision utilisant l'Impureté de Gini."""
    def __init__(self, max_depth=5):
        self.max_depth = max_depth; self.tree = None
    def _gini(self, y):
        if len(y) == 0: return 0
        p = np.bincount(y, minlength=2) / len(y)
        return 1 - np.sum(p**2)
    def _best_split(self, X, y):
        best_g, best_f, best_t = float('inf'), None, None
        for f in range(X.shape[1]):
            for t in np.unique(X[:, f]):
                left = X[:, f] <= t; right = ~left
                if np.sum(left) == 0 or np.sum(right) == 0: continue
                n = len(y)
                g = (np.sum(left)/n)*self._gini(y[left]) + (np.sum(right)/n)*self._gini(y[right])
                if g < best_g: best_g, best_f, best_t = g, f, t
        return best_f, best_t
    def _build(self, X, y, depth=0):
        if len(np.unique(y)) == 1: return int(y[0])
        if depth >= self.max_depth or len(y) < 2: return int(np.round(np.mean(y)))
        f, t = self._best_split(X, y)
        if f is None: return int(np.round(np.mean(y)))
        left = X[:, f] <= t
        if np.sum(left) == 0 or np.sum(~left) == 0: return int(np.round(np.mean(y)))
        return {'f': f, 't': t, 'l': self._build(X[left], y[left], depth+1), 'r': self._build(X[~left], y[~left], depth+1)}
    def fit(self, X, y): self.tree = self._build(X, y)
    def _pred1(self, x, node):
        if not isinstance(node, dict): return node
        return self._pred1(x, node['l'] if x[node['f']] <= node['t'] else node['r'])
    def predict(self, X): return np.array([self._pred1(x, self.tree) for x in X])

class C45FromScratch:
    """Arbre de décision utilisant le Ratio de Gain (amélioration d'ID3)."""
    def __init__(self, max_depth=5):
        self.max_depth = max_depth; self.tree = None
    def _entropy(self, y):
        if len(y) == 0: return 0
        p = np.bincount(y, minlength=2) / len(y)
        return -np.sum([pi * np.log2(pi) for pi in p if pi > 0])
    def _gain_ratio(self, X, y, feat, thr):
        left = X[:, feat] <= thr; right = ~left
        if np.sum(left) == 0 or np.sum(right) == 0: return 0
        n = len(y); pl = np.sum(left)/n; pr = np.sum(right)/n
        ig = self._entropy(y) - pl*self._entropy(y[left]) - pr*self._entropy(y[right])
        si = -pl*np.log2(pl+1e-10) - pr*np.log2(pr+1e-10)
        return ig / si if si > 0 else 0
    def _best_split(self, X, y):
        best_g, best_f, best_t = -1, None, None
        for f in range(X.shape[1]):
            for t in np.unique(X[:, f]):
                g = self._gain_ratio(X, y, f, t)
                if g > best_g: best_g, best_f, best_t = g, f, t
        return best_f, best_t
    def _build(self, X, y, depth=0):
        if len(np.unique(y)) == 1: return int(y[0])
        if depth >= self.max_depth or len(y) < 2: return int(np.round(np.mean(y)))
        f, t = self._best_split(X, y)
        if f is None: return int(np.round(np.mean(y)))
        left = X[:, f] <= t
        if np.sum(left) == 0 or np.sum(~left) == 0: return int(np.round(np.mean(y)))
        return {'f': f, 't': t, 'l': self._build(X[left], y[left], depth+1), 'r': self._build(X[~left], y[~left], depth+1)}
    def fit(self, X, y): self.tree = self._build(X, y)
    def _pred1(self, x, node):
        if not isinstance(node, dict): return node
        return self._pred1(x, node['l'] if x[node['f']] <= node['t'] else node['r'])
    def predict(self, X): return np.array([self._pred1(x, self.tree) for x in X])

# ==========================================
# EXÉCUTION PRINCIPALE
# ==========================================
try:
    print("\n" + "==="*15)
    print("   LANCEMENT DES TESTS DE MACHINE LEARNING")
    print("==="*15 + "\n")

    log_step("Chargement du dataset 'medical_costs.csv'", 0.5)
    try:
        df = pd.read_csv('data/medical_costs.csv')
    except FileNotFoundError:
        print("\n[ERREUR] Fichier introuvable."); sys.exit(1)

    # Données régression
    X_simple = df[['age']].values; y = df[['cost']].values
    X_multi = df[['age', 'bmi', 'children']].values

    # Données classification
    median_cost = np.median(df['cost'].values)
    y_class = (df['cost'].values >= median_cost).astype(int)
    X_class = df[['age', 'bmi']].values
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_cls)
    X_test_scaled = scaler.transform(X_test_cls)

    # Grille pour frontières de décision
    h = 0.5
    x_min, x_max = X_class[:, 0].min() - 2, X_class[:, 0].max() + 2
    y_min, y_max = X_class[:, 1].min() - 2, X_class[:, 1].max() + 2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_scaled = scaler.transform(grid_points)
    class_colors = np.where(y_class == 1, '#e74c3c', '#3498db')
    log_step("Préparation des données", 0.5)

    # Légende classification réutilisable
    def cls_legend(l1, l2):
        return [Patch(facecolor='#d35400', alpha=0.3, label=l1),
                Line2D([0],[0], color='red', linestyle='dashed', linewidth=2, label=l2),
                Patch(facecolor='#e74c3c', label='Coût Élevé'),
                Patch(facecolor='#3498db', label='Coût Faible')]

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.canvas.manager.set_window_title('Aperçu des Modèles de Machine Learning')

    # ====== 1. RÉGRESSION LINÉAIRE SIMPLE ======
    print("\n" + "-"*45 + "\n 1 : RÉGRESSION LINÉAIRE SIMPLE\n" + "-"*45 + '\n')
    log_step("Calcul des modèles (From Scratch vs Bibliothèque)")
    m1 = RegressionFromScratch(); m1.fit(X_simple, y)
    s1 = LinearRegression().fit(X_simple, y)
    X_plot = np.linspace(X_simple.min(), X_simple.max(), 100).reshape(-1, 1)
    axes[0,0].scatter(X_simple, y, color='gray', alpha=0.5, label='Données Réelles')
    axes[0,0].plot(X_plot, m1.predict(X_plot), color='blue', linewidth=4, label='From Scratch')
    axes[0,0].plot(X_plot, s1.predict(X_plot), color='red', linestyle='dashed', linewidth=2, label='Bibliothèque')
    axes[0,0].set_title('Régression Linéaire Simple'); axes[0,0].set_xlabel('Âge'); axes[0,0].set_ylabel('Coût Médical'); axes[0,0].legend()
    plt.figure(); plt.scatter(X_simple, y, color='gray', alpha=0.5)
    plt.plot(X_plot, m1.predict(X_plot), color='blue', linewidth=2)
    plt.title('Régression Linéaire Simple'); plt.xlabel('Âge'); plt.ylabel('Coût Médical')
    plt.savefig('graphs/regression_lineaire_simple.png'); plt.close()
    log_step("Sauvegarde du graphique", 0.5)

    # ====== 2. RÉGRESSION LINÉAIRE MULTIPLE ======
    print("\n" + "-"*45 + "\n 2 : RÉGRESSION LINÉAIRE MULTIPLE\n" + "-"*45 + '\n')
    log_step("Calcul des modèles (From Scratch vs Bibliothèque)")
    m2 = RegressionFromScratch(); m2.fit(X_multi, y)
    s2 = LinearRegression().fit(X_multi, y)
    yp2 = m2.predict(X_multi)
    axes[0,1].scatter(y, yp2, color='green', alpha=0.5, label='Prédictions (From Scratch)')
    axes[0,1].plot([y.min(),y.max()],[y.min(),y.max()], color='black', linestyle='dashed', lw=2, label='Ajustement Parfait')
    axes[0,1].set_title('Régression Multiple (Réel vs Prédit)'); axes[0,1].set_xlabel('Coût Réel'); axes[0,1].set_ylabel('Coût Prédit'); axes[0,1].legend()
    plt.figure(); plt.scatter(y, yp2, color='green', alpha=0.5)
    plt.plot([y.min(),y.max()],[y.min(),y.max()], color='black', linestyle='dashed', lw=2)
    plt.title('Régression Multiple'); plt.xlabel('Coût Réel'); plt.ylabel('Coût Prédit')
    plt.savefig('graphs/regression_multiple.png'); plt.close()
    log_step("Sauvegarde du graphique", 0.5)

    # ====== 3. RÉGRESSION POLYNOMIALE ======
    print("\n" + "-"*45 + "\n 3 : RÉGRESSION POLYNOMIALE\n" + "-"*45 + '\n')
    log_step("Calcul des modèles (From Scratch vs Bibliothèque)")
    m3 = RegressionFromScratch(); m3.fit(np.c_[X_simple, X_simple**2], y)
    pf = PolynomialFeatures(degree=2, include_bias=False); Xpsk = pf.fit_transform(X_simple)
    s3 = LinearRegression().fit(Xpsk, y)
    Xpp = np.c_[X_plot, X_plot**2]; Xppsk = pf.transform(X_plot)
    axes[0,2].scatter(X_simple, y, color='gray', alpha=0.5, label='Données Réelles')
    axes[0,2].plot(X_plot, m3.predict(Xpp), color='purple', linewidth=4, label='From Scratch')
    axes[0,2].plot(X_plot, s3.predict(Xppsk), color='orange', linestyle='dashed', linewidth=2, label='Bibliothèque')
    axes[0,2].set_title('Régression Polynomiale (Degré 2)'); axes[0,2].set_xlabel('Âge'); axes[0,2].set_ylabel('Coût Médical'); axes[0,2].legend()
    plt.figure(); plt.scatter(X_simple, y, color='gray', alpha=0.5)
    plt.plot(X_plot, m3.predict(Xpp), color='purple', linewidth=2)
    plt.title('Régression Polynomiale'); plt.xlabel('Âge'); plt.ylabel('Coût Médical')
    plt.savefig('graphs/regression_polynomiale.png'); plt.close()
    log_step("Sauvegarde du graphique", 0.5)

    # Helper pour tracer classification
    def plot_cls(ax, Z1, Z2, title, l1, l2, save_path):
        ax.contourf(xx, yy, Z1, alpha=0.3, cmap='coolwarm', levels=np.linspace(-0.5,1.5,3))
        ax.contour(xx, yy, Z2, colors='red', linestyles='dashed', linewidths=2.5, levels=[0.5])
        ax.scatter(X_class[:,0], X_class[:,1], c=class_colors, edgecolors='black', s=20, alpha=0.6, linewidths=0.5)
        ax.set_title(title); ax.set_xlabel('Âge'); ax.set_ylabel('BMI')
        ax.legend(handles=cls_legend(l1, l2), fontsize=7, loc='upper left')
        fig2 = plt.figure(figsize=(8,6))
        plt.contourf(xx, yy, Z1, alpha=0.3, cmap='coolwarm', levels=np.linspace(-0.5,1.5,3))
        plt.scatter(X_class[:,0], X_class[:,1], c=class_colors, edgecolors='black', s=20, alpha=0.6, linewidths=0.5)
        plt.title(title); plt.xlabel('Âge'); plt.ylabel('BMI')
        plt.legend(handles=[Patch(facecolor='#e74c3c', label='Coût Élevé'), Patch(facecolor='#3498db', label='Coût Faible')])
        plt.savefig(save_path); plt.close(fig2)

    # ====== 4. RÉGRESSION LOGISTIQUE ======
    print("\n" + "-"*45 + "\n 4 : RÉGRESSION LOGISTIQUE\n" + "-"*45 + '\n')
    log_step("Calcul des modèles (From Scratch vs Bibliothèque)")
    m4 = LogisticRegressionFromScratch(); m4.fit(X_train_scaled, y_train_cls)
    s4 = SklearnLogisticRegression(max_iter=1000); s4.fit(X_train_scaled, y_train_cls)
    Z4a = m4.predict(grid_points_scaled).reshape(xx.shape)
    Z4b = s4.predict(grid_points_scaled).reshape(xx.shape)
    plot_cls(axes[1,0], Z4a, Z4b, 'Régression Logistique', 'From Scratch', 'Bibliothèque', 'graphs/regression_logistique.png')
    log_step("Sauvegarde du graphique", 0.5)

    # ====== 5. KNN ======
    print("\n" + "-"*45 + "\n 5 : KNN (K PLUS PROCHES VOISINS)\n" + "-"*45 + '\n')
    log_step("Calcul des modèles (From Scratch vs Bibliothèque)")
    m5 = KNNFromScratch(k=5); m5.fit(X_train_scaled, y_train_cls)
    s5 = KNeighborsClassifier(n_neighbors=5); s5.fit(X_train_scaled, y_train_cls)
    print("  [*] Calcul des frontières de décision KNN...", end="", flush=True)
    Z5a = m5.predict(grid_points_scaled).reshape(xx.shape)
    Z5b = s5.predict(grid_points_scaled).reshape(xx.shape)
    print(" TERMINÉ")
    plot_cls(axes[1,1], Z5a, Z5b, 'KNN (K=5)', 'From Scratch', 'Bibliothèque', 'graphs/knn.png')
    log_step("Sauvegarde du graphique", 0.5)

    # ====== 6. SVM ======
    print("\n" + "-"*45 + "\n 6 : SVM (MACHINE À VECTEURS DE SUPPORT)\n" + "-"*45 + '\n')
    log_step("Calcul des modèles (Noyau Linéaire vs Noyau RBF)")
    s6a = SVC(kernel='linear'); s6a.fit(X_train_scaled, y_train_cls)
    s6b = SVC(kernel='rbf'); s6b.fit(X_train_scaled, y_train_cls)
    Z6a = s6a.predict(grid_points_scaled).reshape(xx.shape)
    Z6b = s6b.predict(grid_points_scaled).reshape(xx.shape)
    plot_cls(axes[1,2], Z6a, Z6b, 'SVM', 'Noyau Linéaire', 'Noyau RBF', 'graphs/svm.png')
    log_step("Sauvegarde du graphique", 0.5)

    # ====== 7. ID3 ======
    print("\n" + "-"*45 + "\n 7 : ID3 (ARBRE DE DÉCISION - ENTROPIE)\n" + "-"*45 + '\n')
    log_step("Calcul des modèles (From Scratch vs Bibliothèque)")
    m7 = ID3FromScratch(max_depth=5); m7.fit(X_train_cls, y_train_cls)
    s7 = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42); s7.fit(X_train_cls, y_train_cls)
    print("  [*] Calcul des frontières de décision ID3...", end="", flush=True)
    Z7a = m7.predict(grid_points).reshape(xx.shape)
    Z7b = s7.predict(grid_points).reshape(xx.shape)
    print(" TERMINÉ")
    plot_cls(axes[2,0], Z7a, Z7b, 'ID3 (Entropie)', 'From Scratch', 'Bibliothèque', 'graphs/id3.png')
    log_step("Sauvegarde du graphique", 0.5)

    # ====== 8. CART ======
    print("\n" + "-"*45 + "\n 8 : CART (ARBRE DE DÉCISION - GINI)\n" + "-"*45 + '\n')
    log_step("Calcul des modèles (From Scratch vs Bibliothèque)")
    m8 = CARTFromScratch(max_depth=5); m8.fit(X_train_cls, y_train_cls)
    s8 = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42); s8.fit(X_train_cls, y_train_cls)
    print("  [*] Calcul des frontières de décision CART...", end="", flush=True)
    Z8a = m8.predict(grid_points).reshape(xx.shape)
    Z8b = s8.predict(grid_points).reshape(xx.shape)
    print(" TERMINÉ")
    plot_cls(axes[2,1], Z8a, Z8b, 'CART (Gini)', 'From Scratch', 'Bibliothèque', 'graphs/cart.png')
    log_step("Sauvegarde du graphique", 0.5)

    # ====== 9. C4.5 ======
    print("\n" + "-"*45 + "\n 9 : C4.5 (ARBRE DE DÉCISION - RATIO DE GAIN)\n" + "-"*45 + '\n')
    log_step("Calcul des modèles (From Scratch vs Bibliothèque)")
    m9 = C45FromScratch(max_depth=5); m9.fit(X_train_cls, y_train_cls)
    s9 = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42); s9.fit(X_train_cls, y_train_cls)
    print("  [*] Calcul des frontières de décision C4.5...", end="", flush=True)
    Z9a = m9.predict(grid_points).reshape(xx.shape)
    Z9b = s9.predict(grid_points).reshape(xx.shape)
    print(" TERMINÉ")
    plot_cls(axes[2,2], Z9a, Z9b, 'C4.5 (Ratio de Gain)', 'From Scratch', 'Bibliothèque', 'graphs/c45.png')
    log_step("Sauvegarde du graphique", 0.5)

    # ====== FINALISATION ======
    print("\n" + "==="*15)
    print("\n[SUCCÈS] Tous les modèles ont été générés.")
    print("[INFO] Fermez la fenêtre du graphe ou appuyez sur Ctrl+C.")
    fig.suptitle('Aperçu des Modèles de Machine Learning', fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    print("\n[*] Sauvegarde du graphique combiné final...")
    fig.savefig('graphs/graphe_combine_final.png', dpi=150, bbox_inches='tight')
    print("[*] Fin du programme avec succès.")

except KeyboardInterrupt:
    print("\n\n [ATTENTION] Interruption clavier détectée (Ctrl+C).")
    try:
        fig.savefig('graphs/graphe_combine_final.png', dpi=150, bbox_inches='tight')
        print(" [*] Sauvegarde réussie.")
    except: pass
    sys.exit(0)
