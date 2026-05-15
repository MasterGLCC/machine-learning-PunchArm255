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
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.ensemble import GradientBoostingClassifier

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

class DBSCANFromScratch:
    """DBSCAN implémenté manuellement."""
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps; self.min_samples = min_samples; self.labels_ = None
    def _region_query(self, X, idx):
        return np.where(np.sqrt(np.sum((X - X[idx])**2, axis=1)) <= self.eps)[0]
    def fit_predict(self, X):
        n = len(X); self.labels_ = np.full(n, -1); cluster_id = 0; visited = np.zeros(n, dtype=bool)
        for i in range(n):
            if visited[i]: continue
            visited[i] = True; neighbors = self._region_query(X, i)
            if len(neighbors) < self.min_samples: continue
            self.labels_[i] = cluster_id; seeds = list(neighbors); j = 0
            while j < len(seeds):
                q = seeds[j]
                if not visited[q]:
                    visited[q] = True; q_nb = self._region_query(X, q)
                    if len(q_nb) >= self.min_samples: seeds.extend(q_nb)
                if self.labels_[q] == -1: self.labels_[q] = cluster_id
                j += 1
            cluster_id += 1
        return self.labels_

class NaiveBayesFromScratch:
    """Naive Bayes Gaussien implémenté manuellement."""
    def __init__(self): self.mean = None; self.var = None; self.priors = None; self.classes = None
    def fit(self, X, y):
        self.classes = np.unique(y); nc = len(self.classes); nf = X.shape[1]
        self.mean = np.zeros((nc, nf)); self.var = np.zeros((nc, nf)); self.priors = np.zeros(nc)
        for i, c in enumerate(self.classes):
            Xc = X[y == c]; self.mean[i] = Xc.mean(0); self.var[i] = Xc.var(0); self.priors[i] = len(Xc)/len(y)
    def predict(self, X):
        preds = []
        for x in X:
            posts = []
            for i, c in enumerate(self.classes):
                prior = np.log(self.priors[i])
                lk = -0.5*np.sum(np.log(2*np.pi*self.var[i]+1e-10) + (x-self.mean[i])**2/(self.var[i]+1e-10))
                posts.append(prior + lk)
            preds.append(self.classes[np.argmax(posts)])
        return np.array(preds)

class PCAFromScratch:
    """Analyse en Composantes Principales implémentée manuellement."""
    def __init__(self, n_components=2): self.n_components = n_components; self.components = None; self.mean = None
    def fit(self, X):
        self.mean = np.mean(X, axis=0); Xc = X - self.mean
        cov = np.cov(Xc.T); eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]; self.components = eigvecs[:, idx[:self.n_components]]
        self.explained_variance_ratio_ = eigvals[idx[:self.n_components]] / np.sum(eigvals)
        return self
    def transform(self, X): return (X - self.mean).dot(self.components)

class XGBoostFromScratch:
    """Gradient Boosting simplifié avec souches de décision."""
    def __init__(self, n_estimators=50, lr=0.1):
        self.n_est = n_estimators; self.lr = lr; self.stumps = []; self.base = 0
    def _sigmoid(self, z): return 1/(1+np.exp(-np.clip(z,-500,500)))
    def fit(self, X, y):
        pos = np.sum(y==1); neg = np.sum(y==0)
        self.base = np.log((pos+1e-10)/(neg+1e-10))
        F = np.full(len(y), self.base, dtype=float)
        for _ in range(self.n_est):
            res = y - self._sigmoid(F); best = (0, 0, 0, 0, float('inf'))
            for f in range(X.shape[1]):
                for t in np.percentile(X[:,f], np.arange(10,100,10)):
                    lm = X[:,f]<=t
                    if np.sum(lm)==0 or np.sum(~lm)==0: continue
                    lv, rv = np.mean(res[lm]), np.mean(res[~lm])
                    loss = np.mean((res - np.where(lm, lv, rv))**2)
                    if loss < best[4]: best = (f, t, lv, rv, loss)
            self.stumps.append(best[:4])
            f,t,lv,rv = best[:4]; F += self.lr * np.where(X[:,f]<=t, lv, rv)
    def predict(self, X):
        F = np.full(len(X), self.base, dtype=float)
        for f,t,lv,rv in self.stumps: F += self.lr * np.where(X[:,f]<=t, lv, rv)
        return (self._sigmoid(F) >= 0.5).astype(int)

class GridWorld:
    """Environnement GridWorld pour le Q-Learning."""
    def __init__(self, size=5):
        self.size = size; self.goal = (size-1, size-1)
        self.obstacles = [(1,1),(2,3),(3,1)]; self.actions = [(0,1),(0,-1),(1,0),(-1,0)]
    def reset(self): self.state = (0,0); return self.state
    def step(self, action):
        dx, dy = self.actions[action]
        ns = (max(0,min(self.size-1,self.state[0]+dx)), max(0,min(self.size-1,self.state[1]+dy)))
        if ns in self.obstacles: ns = self.state
        self.state = ns
        if self.state == self.goal: return self.state, 10, True
        return self.state, -0.1, False

class QLearningAgent:
    """Agent Q-Learning."""
    def __init__(self, size=5, lr=0.1, gamma=0.99, epsilon=0.1):
        self.q = np.zeros((size, size, 4)); self.lr = lr; self.gamma = gamma; self.eps = epsilon
    def act(self, s):
        if np.random.random() < self.eps: return np.random.randint(4)
        return np.argmax(self.q[s[0], s[1]])
    def update(self, s, a, r, ns):
        self.q[s[0],s[1],a] += self.lr*(r + self.gamma*np.max(self.q[ns[0],ns[1]]) - self.q[s[0],s[1],a])

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
    X_all = df[['age', 'bmi', 'children']].values  # Pour PCA

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

    fig, axes = plt.subplots(5, 3, figsize=(18, 30))
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

    # ====== 10. DBSCAN ======
    print("\n" + "-"*45 + "\n 10 : DBSCAN (CLUSTERING)\n" + "-"*45 + '\n')
    log_step("Calcul des modèles (From Scratch vs Bibliothèque)")
    db_scratch = DBSCANFromScratch(eps=0.8, min_samples=5)
    labels_scratch = db_scratch.fit_predict(X_train_scaled)
    db_sklearn = SklearnDBSCAN(eps=0.8, min_samples=5)
    labels_sklearn = db_sklearn.fit_predict(X_train_scaled)
    axes[3,0].scatter(X_train_cls[:,0], X_train_cls[:,1], c=labels_scratch, cmap='tab10', s=20, alpha=0.7, edgecolors='black', linewidths=0.5)
    axes[3,0].set_title(f'DBSCAN From Scratch ({len(set(labels_scratch)-{-1})} clusters)')
    axes[3,0].set_xlabel('Âge'); axes[3,0].set_ylabel('BMI')
    fig_db = plt.figure(figsize=(8,6))
    plt.scatter(X_train_cls[:,0], X_train_cls[:,1], c=labels_scratch, cmap='tab10', s=20, alpha=0.7, edgecolors='black', linewidths=0.5)
    plt.title('DBSCAN (From Scratch)'); plt.xlabel('Âge'); plt.ylabel('BMI')
    plt.savefig('graphs/dbscan.png'); plt.close(fig_db)
    log_step("Sauvegarde du graphique", 0.5)

    # ====== 11. NAIVE BAYES ======
    print("\n" + "-"*45 + "\n 11 : NAIVE BAYES\n" + "-"*45 + '\n')
    log_step("Calcul des modèles (From Scratch vs Bibliothèque)")
    m11 = NaiveBayesFromScratch(); m11.fit(X_train_scaled, y_train_cls)
    s11 = GaussianNB(); s11.fit(X_train_scaled, y_train_cls)
    Z11a = m11.predict(grid_points_scaled).reshape(xx.shape)
    Z11b = s11.predict(grid_points_scaled).reshape(xx.shape)
    plot_cls(axes[3,1], Z11a, Z11b, 'Naive Bayes', 'From Scratch', 'Bibliothèque', 'graphs/naive_bayes.png')
    log_step("Sauvegarde du graphique", 0.5)

    # ====== 12. PCA ======
    print("\n" + "-"*45 + "\n 12 : PCA (RÉDUCTION DE DIMENSIONS)\n" + "-"*45 + '\n')
    log_step("Calcul des modèles (From Scratch vs Bibliothèque)")
    pca_scratch = PCAFromScratch(n_components=2); pca_scratch.fit(X_all)
    X_pca_scratch = pca_scratch.transform(X_all)
    pca_sklearn = SklearnPCA(n_components=2); pca_sklearn.fit(X_all)
    X_pca_sklearn = pca_sklearn.transform(X_all)
    colors_pca = np.where(y_class == 1, '#e74c3c', '#3498db')
    axes[3,2].scatter(X_pca_scratch[:,0], X_pca_scratch[:,1], c=colors_pca, s=20, alpha=0.6, edgecolors='black', linewidths=0.5)
    axes[3,2].set_title('PCA (From Scratch)'); axes[3,2].set_xlabel('Composante 1'); axes[3,2].set_ylabel('Composante 2')
    axes[3,2].legend(handles=[Patch(facecolor='#e74c3c', label='Coût Élevé'), Patch(facecolor='#3498db', label='Coût Faible')], fontsize=7)
    fig_pca = plt.figure(figsize=(8,6))
    plt.scatter(X_pca_scratch[:,0], X_pca_scratch[:,1], c=colors_pca, s=20, alpha=0.6, edgecolors='black', linewidths=0.5)
    plt.title('PCA — Projection 2D'); plt.xlabel('Composante 1'); plt.ylabel('Composante 2')
    plt.legend(handles=[Patch(facecolor='#e74c3c', label='Coût Élevé'), Patch(facecolor='#3498db', label='Coût Faible')])
    plt.savefig('graphs/pca.png'); plt.close(fig_pca)
    log_step("Sauvegarde du graphique", 0.5)

    # ====== 13. XGBOOST ======
    print("\n" + "-"*45 + "\n 13 : XGBOOST (GRADIENT BOOSTING)\n" + "-"*45 + '\n')
    log_step("Calcul des modèles (From Scratch vs Bibliothèque)")
    m13 = XGBoostFromScratch(n_estimators=50, lr=0.1); m13.fit(X_train_cls, y_train_cls)
    s13 = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42); s13.fit(X_train_cls, y_train_cls)
    print("  [*] Calcul des frontières de décision XGBoost...", end="", flush=True)
    Z13a = m13.predict(grid_points).reshape(xx.shape)
    Z13b = s13.predict(grid_points).reshape(xx.shape)
    print(" TERMINÉ")
    plot_cls(axes[4,0], Z13a, Z13b, 'XGBoost', 'From Scratch', 'Bibliothèque', 'graphs/xgboost.png')
    log_step("Sauvegarde du graphique", 0.5)

    # ====== 14. Q-LEARNING ======
    print("\n" + "-"*45 + "\n 14 : Q-LEARNING (APPRENTISSAGE PAR RENFORCEMENT)\n" + "-"*45 + '\n')
    log_step("Entraînement de l'agent Q-Learning")
    env = GridWorld(size=5); agent = QLearningAgent(size=5)
    for ep in range(1000):
        s = env.reset(); done = False
        while not done:
            a = agent.act(s); ns, r, done = env.step(a); agent.update(s, a, r, ns); s = ns
    arrows = ['→','←','↓','↑']; best_actions = np.argmax(agent.q, axis=2)
    max_q = np.max(agent.q, axis=2)
    axes[4,1].imshow(max_q, cmap='YlOrRd', interpolation='nearest')
    for i in range(5):
        for j in range(5):
            if (i,j) == env.goal: axes[4,1].text(j, i, '★', ha='center', va='center', fontsize=16)
            elif (i,j) in env.obstacles: axes[4,1].text(j, i, '■', ha='center', va='center', fontsize=14, color='black')
            else: axes[4,1].text(j, i, arrows[best_actions[i,j]], ha='center', va='center', fontsize=14, color='white')
    axes[4,1].set_title('Q-Learning (Politique Apprise)'); axes[4,1].set_xticks([]); axes[4,1].set_yticks([])
    fig_ql = plt.figure(figsize=(6,6))
    plt.imshow(max_q, cmap='YlOrRd', interpolation='nearest')
    for i in range(5):
        for j in range(5):
            if (i,j) == env.goal: plt.text(j, i, '★', ha='center', va='center', fontsize=20)
            elif (i,j) in env.obstacles: plt.text(j, i, '■', ha='center', va='center', fontsize=18, color='black')
            else: plt.text(j, i, arrows[best_actions[i,j]], ha='center', va='center', fontsize=18, color='white')
    plt.title('Q-Learning — Politique Apprise'); plt.xticks([]); plt.yticks([]); plt.colorbar(label='Q-Value Max')
    plt.savefig('graphs/qlearning.png'); plt.close(fig_ql)
    axes[4,2].set_visible(False)  # Cacher la case vide
    log_step("Sauvegarde du graphique", 0.5)

    # ====== FINALISATION ======
    print("\n" + "==="*15)
    print("\n[SUCCÈS] Tous les modèles ont été générés.")
    print("[INFO] Fermez la fenêtre du graphe ou appuyez sur Ctrl+C.")
    fig.suptitle('Aperçu des Modèles de Machine Learning', fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
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
