# Contexte du projet

Vous êtes développeur IA junior au sein d'un laboratoire biomédical
spécialisé en imagerie médicale. Le laboratoire souhaite développer une
solution d'analyse automatisée d'images cérébrales afin d'assister les
spécialistes dans l'identification et la localisation des tumeurs
cérébrales.

Votre mission consiste à concevoir une solution unifiée basée sur le
Deep Learning, intégrant deux approches complémentaires :

-   La classification d'images cérébrales à l'aide d'un modèle
    pré-entraîné basé sur GoogLeNet et TensorFlow/Keras.
-   La détection et la localisation des tumeurs cérébrales à l'aide de
    YOLOv8.

Vous devrez également mettre en place un système de suivi des
expériences, automatiser les étapes du pipeline et exposer les modèles à
travers des API REST, consommées par une interface Streamlit.

# Classification des tumeurs cérébrales avec GoogleNet

-   Charger les images, vérifier extensions et intégrité (try-except),
    écarter les fichiers invalides.
-   Explorer les classes (noms de dossiers), countplot, afficher un
    échantillon par classe.
-   Répartir les images en trois ensembles : Train (70 %), Validation
    (15 %) et Test (15 %), puis compter et afficher le nombre d'images
    dans chaque ensemble.
-   Appliquer des techniques d'augmentation des données, telles que le
    flou (blur), l'ajout de bruit (noise) et le retournement (flip),
    uniquement sur les données d'entraînement, afin d'enrichir les
    échantillons et de réduire les déséquilibres entre les classes.
-   Charger avec `imagedatasetfrom_directory()` (redimensionnement,
    batch, `shuffle=True` pour le train), puis normaliser selon le
    modèle.
-   Charger le modèle préentraîné GoogLeNet (InceptionV1), geler ses
    couches de base, remplacer sa tête de classification par une
    nouvelle adaptée au nombre de classes du dataset, puis entraîner
    uniquement cette dernière pour effectuer le Feature Extraction.
-   Déterminer les hyperparamètres d'entraînement, notamment le learning
    rate, la fonction de perte (loss function) et l'optimiseur
    (optimizer).
-   Entraîner le modèle en utilisant l'ensemble d'entraînement et suivre
    ses performances sur l'ensemble de validation.
-   Évaluer le modèle à l'aide de métriques adaptées, notamment
    l'accuracy, la précision (precision), le rappel (recall) et le
    F1-score.
-   Analyser les résultats à l'aide d'une matrice de confusion et
    comparer les performances sur les ensembles d'entraînement et de
    test.
-   Dégeler les dernières couches du modèle préentraîné, poursuivre le
    fine-tuning avec un faible learning rate, puis réévaluer le modèle
    et comparer ses performances avec celles du modèle obtenu après le
    Feature Extraction.
-   Sauvegarder le modèle entraîné.

# Détection des tumeurs cérébrales avec YOLOv8

-   Afficher un échantillon d'images avec leurs boîtes englobantes
    (bounding boxes) et les classes correspondantes.
-   Vérifier, pour chaque image, l'existence d'un fichier `.txt`
    correspondant dans le répertoire des annotations, en comparant les
    noms de base des fichiers.
-   Organiser les données dans les dossiers images et labels, en
    respectant les ensembles train, valid et test.
-   Copier chaque image accompagnée de son fichier d'annotation
    correspondant vers l'ensemble approprié.
-   Écarter les images sans annotation correspondante et les fichiers
    d'annotation sans image associée.
-   Vérifier la validité du format des annotations YOLO et la cohérence
    des classes.
-   Compter les images et les fichiers d'annotation présents dans chaque
    ensemble.
-   Créer un fichier `data.yaml` contenant les chemins des ensembles
    d'entraînement, de validation et de test, le nombre de classes et
    leurs noms.
-   Définir les paramètres d'augmentation des données dans la
    configuration d'entraînement de YOLOv8.
-   Entraîner le modèle YOLOv8 en choisissant des hyperparamètres
    adaptés au dataset et aux ressources disponibles.
-   Évaluer les performances à l'aide des métriques adaptées à la
    détection d'objets, notamment la précision, le rappel, la mAP@0.5 et
    la mAP@0.5:0.95.
-   Analyser les résultats et vérifier la capacité de généralisation du
    modèle sur l'ensemble de test.
-   Sauvegarder le modèle entraîné et les artefacts nécessaires à son
    utilisation.

# Suivi des expériences avec MLflow

-   Configurer une expérience MLflow pour suivre les entraînements.
-   Enregistrer les hyperparamètres utilisés, les métriques obtenues et
    les artefacts pertinents.
-   Comparer plusieurs expériences afin d'identifier la configuration la
    plus performante.
-   Enregistrer le modèle retenu et conserver les informations
    nécessaires à son identification et à sa réutilisation.

# Automatisation du pipeline avec Apache Airflow

-   Créer un DAG Airflow représentant le pipeline de traitement des
    données et d'entraînement.
-   Organiser les tâches dans un ordre logique :
    1.  Vérification et préparation des datasets.
    2.  Validation de la structure et de la qualité des données.
    3.  Entraînement du modèle de classification.
    4.  Entraînement du modèle de détection.
    5.  Évaluation des modèles et enregistrement des résultats dans
        MLflow.
    6.  Sauvegarde des modèles et des artefacts produits.
-   Configurer la gestion des erreurs, les journaux d'exécution (logs)
    et le suivi des statuts des tâches.

# Interface utilisateur avec Streamlit

-   Développer une interface Streamlit permettant de téléverser une
    image cérébrale.
-   Proposer deux fonctionnalités distinctes :
    -   Classification de l'image cérébrale.
    -   Détection et localisation des tumeurs cérébrales.
-   Connecter Streamlit aux endpoints FastAPI à l'aide de requêtes HTTP.
-   Afficher la classe prédite, les scores de confiance et les résultats
    de classification.
-   Afficher l'image avec les boîtes englobantes et les informations de
    détection retournées par YOLOv8.
