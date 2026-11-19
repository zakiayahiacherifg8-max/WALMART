# 🛒 Walmart – Prévision des ventes hebdomadaires

> Analyse exploratoire · Modélisation prédictive · Recommandations business

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Regression-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458)
![Statut](https://img.shields.io/badge/Projet-Bloc%203%20RNCP%20CDSD-green)

**Auteure :** Zakia YAHIA CHERIF
**Contexte :** Projet Data Analytics & Machine Learning – Bloc 3 RNCP CDSD

---

## 📌 Résumé

Ce projet vise à **prédire les ventes hebdomadaires (`Weekly_Sales`) de magasins Walmart** à partir de données historiques et d'indicateurs économiques (chômage, CPI, prix du carburant, température) et temporels.

Trois modèles de régression linéaire sont comparés : **OLS** (référence), **Ridge** et **Lasso**, avec optimisation des hyperparamètres par `GridSearchCV`. Le projet se conclut par une lecture critique des résultats et des recommandations business destinées aux équipes marketing et opérationnelles.

**Résultat principal :** les modèles linéaires atteignent un **R² test proche de 0,95**. Le **magasin** est le facteur le plus structurant, devant la saisonnalité et les variables économiques.

---

## 🎯 Problématique

> Comment prédire les ventes hebdomadaires à partir de données historiques et d'indicateurs économiques, et quels facteurs influencent le plus ces ventes ?

| | |
|---|---|
| **Cible** | `Weekly_Sales` |
| **Type de problème** | Régression supervisée |
| **Utilisateurs** | Équipes marketing et opérationnelles (planification, stocks, actions commerciales) |
| **Décision attendue** | Identifier les facteurs les plus associés aux ventes et connaître les limites de la prévision |

Questions posées dans le notebook :
1. Quelles variables influencent les ventes ?
2. Peut-on prédire les ventes avec précision ?
3. Quel modèle est le plus robuste et généralisable ?

---

## 📂 Données

Jeu de données `Walmart_Store_sales.csv` : **150 observations brutes**, 8 variables (issu d'une compétition Kaggle, modifié pour l'exercice).

| Variable | Description |
|---|---|
| `Store` | Numéro du magasin (20 magasins) |
| `Date` | Date de la semaine |
| `Weekly_Sales` | **Variable cible** : ventes hebdomadaires ($) |
| `Holiday_Flag` | Semaine fériée (1) ou non (0) |
| `Temperature` | Température (°F) |
| `Fuel_Price` | Prix du carburant ($) |
| `CPI` | Indice des prix à la consommation |
| `Unemployment` | Taux de chômage (%) |

**Qualité des données :** des valeurs manquantes sont présentes dans presque toutes les colonnes (entre 12 et 18 par colonne, hors `Store`), et des valeurs atypiques ont été détectées sur le chômage.

---

## 🔧 Préparation des données

1. **Suppression des lignes sans valeur cible** (aucune imputation sur la cible, pour éviter de biaiser les prédictions).
2. **Variables temporelles** créées à partir de `Date` : année, mois, jour, jour de la semaine.
3. **Valeurs atypiques :** règle des 3 écarts-types → **5 valeurs aberrantes** identifiées sur `Unemployment`.
4. **Pipeline scikit-learn** (`ColumnTransformer`) :
   - Variables numériques : `SimpleImputer` (médiane) + `StandardScaler`
   - Variables catégorielles (`Store`, jour de la semaine) : `SimpleImputer` (valeur la plus fréquente) + `OneHotEncoder`
5. Après nettoyage : **environ 93 observations exploitables**.

---

## 🔍 Analyse exploratoire (EDA) – principaux constats

- **Le magasin domine les ventes.** Les ventes moyennes varient fortement d'un magasin à l'autre ; certains vendent près de deux fois plus que d'autres.
- **Saisonnalité marquée :** pic en novembre–décembre (Black Friday, Noël), creux en janvier–février, soit environ **+20 % en décembre par rapport à janvier**. Un pic secondaire, plus modéré, apparaît en été.
- **Jours fériés :** environ **+7,3 %** de ventes en moyenne sur les semaines fériées (seulement 8 % des semaines sont fériées).
- **Variables économiques :** corrélations modestes avec les ventes ; le **CPI** est la plus marquée (**≈ −0,29**). Une corrélation CPI ↔ chômage (≈ −0,35) suggère une possible **multicolinéarité**.
- **Température :** effet négatif modéré sur les ventes.

> ⚠️ Ce sont des associations observées, pas des preuves de causalité.

---

## 🤖 Modélisation

Trois approches sont comparées sur le même jeu de test :

| Modèle | Principe |
|---|---|
| **Régression linéaire (OLS)** | Modèle de référence |
| **Ridge** | Régularisation L2, `alpha` ajusté par validation croisée / `GridSearchCV` |
| **Lasso** | Régularisation L1, sélection potentielle de variables, **modèle final retenu** |

Optimisation des hyperparamètres par **validation croisée à 5 plis** pour Ridge et Lasso.

### Résultats (jeu de test)

| Modèle | R² test | MAE | RMSE |
|---|---|---|---|
| **Régression linéaire (OLS)** | **0,9528** | **127 441 $** | **147 284 $** |
| Ridge – GridSearch (α ≈ 0,038) | 0,9466 | 135 775 $ | 156 754 $ |
| Lasso – GridSearch (α = 1000) | 0,9466 | 134 649 $ | 156 708 $ |

**Lecture des résultats :**
- L'**OLS obtient les meilleures métriques brutes**, mais ses coefficients sont instables.
- **Ridge et Lasso atteignent le même R² test (0,9466)**, ce qui confirme leur équivalence sur ces données.
- **Lasso est retenu** pour son compromis simplicité / performance et des coefficients plus crédibles : il élimine 2 magasins peu distinctifs, réduit l'effet du CPI d'environ 35 % et révèle que `Holiday_Flag` a peu d'effet réel.
- Le modèle final conserve **24 variables sur 26**.

### Modèle final

`Lasso(alpha=1000, max_iter=10000)`

| Indicateur | Valeur |
|---|---|
| R² test | 0,9466 |
| MAE | 134 649 $ |
| Erreur relative moyenne | 15,1 % |
| Précision (100 − erreur relative) | 84,9 % |

Le modèle prédit très bien les **gros magasins** (> 1,5 M$ de ventes) et est moins précis sur les petits volumes (0,25–0,5 M$). La médiane des erreurs est proche de 0 (pas de biais systématique), avec quelques cas extrêmes autour de −40 % à investiguer.

> 💡 **R² élevé ≠ absence d'erreurs :** le RMSE du modèle final reste d'environ 156 700 $.

---

## 🧠 Interprétation & limites

**Insights**
- Le **magasin** est le facteur le plus structurant : le top 5 des coefficients est exclusivement composé de magasins (Store_4 : +63,2 %, Store_14 : +43,2 %, Store_13 : +42,9 %…).
- La **saisonnalité** contribue clairement aux variations de ventes.
- Les **variables économiques seules** expliquent peu.

**Points de vigilance**
- **Échantillon limité** (≈ 93 observations exploitables) : les résultats restent exploratoires.
- **CPI au signe contre-intuitif** dans les modèles (effet positif), probablement dû à la colinéarité : à ne pas interpréter de façon causale.
- **Alpha Lasso très élevé (1000)** : suggère une certaine instabilité, les données résistant à la régularisation.
- **Réduction de dimension limitée** (24 variables conservées sur 26).
- **Données absentes :** promotions, prix produits, superficie des magasins et autres facteurs opérationnels.
- Corrélation ≠ causalité ; résultats à valider sur un échantillon plus large et plus récent avant toute extrapolation.

---

## 💼 Recommandations business

1. **Performance des magasins :** comparer les magasins les plus performants aux moins performants et identifier les pratiques transférables (assortiment, disponibilité, organisation, zone de chalandise).
2. **Planification :** anticiper le pic de novembre–décembre en adaptant stocks, équipes et capacités logistiques.
3. **Pilotage :** utiliser les prévisions comme **aide à la décision**, pas comme vérité absolue ; suivre les erreurs par magasin et investiguer les écarts importants.

> À vérifier avant déploiement : les écarts cités pour certains magasins (ex. Store_5 vs Store_4) doivent être confirmés sur des données métier complètes.

---

## 🚀 Pistes d'amélioration

- Collecter **davantage d'historique** et intégrer promotions, prix, événements locaux et caractéristiques des magasins.
- Valider avec une **séparation temporelle** plutôt qu'un split aléatoire uniquement.
- Comparer à des **modèles non linéaires** (Random Forest, Gradient Boosting).
- Tester des **features d'interaction** (ex. Store × Mois).
- Mesurer **MAE / RMSE par magasin et par période**.

---

## 🗂️ Structure du dépôt

```
├── Analyse_WALMART.ipynb        # Notebook complet : EDA, preprocessing, modèles
├── Walmart_Store_sales.csv      # Jeu de données
├── Presentation_Walmart.pdf     # Support de présentation
└── README.md
```

## ▶️ Exécuter le projet

```bash
# 1. Cloner le dépôt
git clone <url-du-depot>
cd <nom-du-depot>

# 2. Installer les dépendances
pip install pandas numpy matplotlib seaborn scikit-learn plotly jupyter

# 3. Lancer le notebook
jupyter notebook Analyse_WALMART.ipynb
```

## 🛠️ Stack technique

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `scikit-learn` (Pipeline, ColumnTransformer, LinearRegression, Ridge, Lasso, RidgeCV, LassoCV, GridSearchCV)

---

## 👩‍💻 Auteure

**Zakia YAHIA CHERIF** – Data Analyst
Projet réalisé dans le cadre du **Bloc 3 – RNCP CDSD**.
