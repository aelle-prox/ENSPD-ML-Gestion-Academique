# ENSPD-ML-Gestion-Academique
Projet de machine learning ENSPD Doula
# ENSPD - Optimisation de la gestion académique et des ressources

## Description du projet
Ce projet vise à développer un système de **Machine Learning** capable d’améliorer la gestion académique et l’utilisation des ressources au sein de l’École Nationale Supérieure Polytechnique de Douala (ENSPD).  

L’objectif principal est de **prédire la réussite académique des étudiants** en fonction de données telles que leurs notes, leur assiduité et leur participation aux projets. Le système doit également identifier les facteurs clés influençant la performance des étudiants et proposer des recommandations pour optimiser l’allocation des salles, des encadreurs et des équipements.

---

## Données utilisées
Les données sont soigneusement anonymisées pour respecter la confidentialité et incluent :  
- Résultats scolaires  
- Taux de présence  
- Informations sur les filières  
- Disponibilités des infrastructures  

Elles sont enrichies par des variables contextuelles comme le calendrier académique ou les effectifs par promotion. Les données sont collectées et structurées dans des formats exploitables tels que **CSV** ou **Excel**, puis documentées pour faciliter la compréhension.

---

## Prétraitement et feature engineering
Les étapes de préparation des données comprennent :  
- Nettoyage des valeurs manquantes  
- Normalisation des notes  
- Création de nouvelles variables pertinentes (ex : ratio présence/note, moyenne par filière)  

Ces transformations permettent d’améliorer la qualité des modèles et la robustesse des analyses.

---

## Modélisation
Le projet utilise plusieurs approches de Machine Learning :  

1. **Modèles supervisés** :  
   - Régression logistique  
   - Arbres de décision  
   - Évaluation via précision, rappel et F1-score  

   

---

## Visualisation et évaluation
Les résultats sont présentés à l’aide de :  
- Heatmaps  
- Courbes ROC  
- Comparaison des performances des modèles  

---

## Livrables
Le projet final inclut :  
- **Code source documenté**  
- **Dataset académique anonymisé**  
- **Rapport PDF** détaillant la méthodologie et les résultats  
- **Présentation PowerPoint** (20 à 25 diapositives)

---

## Objectif
Fournir à l’administration des outils concrets pour :  
- Anticiper les besoins en encadrement  
- Suivre les étudiants à risque  
- Optimiser l’utilisation des infrastructures  

---

## Technologies et outils
- Python (pandas, numpy, scikit-learn, matplotlib, seaborn)  
- Jupyter Notebook ou IDE Python  
- Git et GitHub pour le versioning

