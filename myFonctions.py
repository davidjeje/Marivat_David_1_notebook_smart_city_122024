import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def display_boxplot_with_stats(dataframe, column_name):
    """
    Affiche un graphique box plot pour une colonne donnée d'un DataFrame
    et ajoute des lignes pour la moyenne, la médiane et l'écart-type.
    
    Parameters:
        dataframe (pd.DataFrame): Le DataFrame contenant les données.
        column_name (str): Le nom de la colonne à afficher dans le boxplot.
    """
    # Calcul des statistiques
    mean = dataframe[column_name].mean()
    median = dataframe[column_name].median()
    std = dataframe[column_name].std()

    # Box plot
    dataframe.boxplot(column=column_name)

    # Ajouter la ligne de la moyenne
    plt.axhline(y=mean, color='r', linestyle='--', label='Moyenne')

    # Ajouter la ligne de la médiane
    plt.axhline(y=median, color='b', linestyle='-', label='Médiane')

    # Ajouter les lignes de l'écart-type
    plt.axhline(y=mean + std, color='g', linestyle=':', label='Écart-type (+)')
    plt.axhline(y=mean - std, color='g', linestyle=':', label='Écart-type (-)')

    # Ajouter des annotations textuelles avec des positions ajustées
    # Moyenne (au-dessus)
    plt.annotate(f'Moyenne: {mean:.2f}', 
                 xy=(0.5, mean + 0.05), xycoords='data',  # Légèrement au-dessus de la ligne
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor='red', facecolor='white'),
                 color='red', fontsize=10, ha='center')

    # Médiane (au-dessous)
    plt.annotate(f'Médiane: {median:.2f}', 
                 xy=(0.85, median), xycoords='data',  # Légèrement en dessous de la ligne
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor='blue', facecolor='white'),
                 color='blue', fontsize=10, ha='center')

    # Écart-type (à droite)
    plt.annotate(f'Écart-type: {std:.2f}', 
                 xy=(1.15, mean + std), xycoords='data',  # À droite de la ligne
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor='green', facecolor='white'),
                 color='green', fontsize=10, ha='left')

    # Titre et labels
    plt.title(f'Box Plot {column_name}')
    plt.ylabel(column_name)

    # Ajouter une légende
    plt.legend()

    # Ajuster la mise en page
    plt.tight_layout()

    # Afficher le graphique
    plt.show()

def tableOfData(dataTree, effectifs, modalites, variable):
    """
    Crée un tableau de fréquences détaillé pour une variable donnée.

    Paramètres :
    - dataTree : DataFrame contenant les données initiales.
    - effectifs : Série Pandas contenant les fréquences relatives des valeurs uniques de la variable.
    - modalites : Index contenant les valeurs uniques (modalités) de la variable.
    - variable : Nom de la colonne analysée (chaîne de caractères).

    Retourne :
    - Un DataFrame avec trois colonnes :
        - [variable] : Les modalités uniques de la variable étudiée.
        - "n" : Les effectifs (fréquences relatives) de chaque modalité.
        - "f" : La fréquence relative ajustée en fonction du nombre total d'observations.
    """

    # Création d'un DataFrame avec les valeurs uniques de la variable
    idv = pd.DataFrame(modalites, columns=[variable])

    # Ajout de la colonne "n" contenant les fréquences relatives issues de `effectifs`
    idv["n"] = effectifs.values

    # Calcul de la fréquence relative "f" en divisant chaque effectif par la taille du DataFrame
    idv["f"] = idv["n"] / len(dataTree)

    # La ligne suivante est commentée, mais elle pourrait être utilisée pour calculer la fréquence cumulée.
    # idv["F"] = idv["f"].cumsum()

    return idv

def barChartTree(dataframe, x_column, y_column, 
                 title="Bar Chart", xlabel=None, ylabel=None, 
                 rotation=90, grid=True, figsize=(12, 6), top_n=None):
    """
    Génère un diagramme en barres à partir d'un DataFrame.

    Paramètres :
    - dataframe : DataFrame contenant les données à représenter.
    - x_column : Nom de la colonne pour l'axe des X (catégories).
    - y_column : Nom de la colonne pour l'axe des Y (valeurs à afficher).
    - title : (optionnel) Titre du graphique (par défaut "Bar Chart").
    - xlabel : (optionnel) Nom personnalisé de l'axe des X (par défaut, nom de `x_column`).
    - ylabel : (optionnel) Nom personnalisé de l'axe des Y (par défaut, nom de `y_column`).
    - rotation : (optionnel) Angle de rotation des étiquettes de l'axe X (90° par défaut).
    - grid : (optionnel) Afficher une grille horizontale si True (activé par défaut).
    - figsize : (optionnel) Taille de la figure (12x6 par défaut).
    - top_n : (optionnel) Nombre maximum de catégories à afficher (les `top_n` plus grandes valeurs selon `y_column`).

    Retour :
    - Affiche un graphique en barres avec les options spécifiées.
    """

    # Optionnel : Filtrer pour ne conserver que les `top_n` valeurs les plus élevées de `y_column`
    if top_n:
        dataframe = dataframe.nlargest(top_n, y_column)

    # Créer une nouvelle figure avec la taille spécifiée
    plt.figure(figsize=figsize)

    # Tracer le diagramme en barres
    plt.bar(dataframe[x_column], dataframe[y_column])

    # Ajouter un titre au graphique
    plt.title(title)

    # Ajouter un libellé à l'axe des X (utilise `xlabel` si fourni, sinon `x_column`)
    plt.xlabel(xlabel if xlabel else x_column)

    # Ajouter un libellé à l'axe des Y (utilise `ylabel` si fourni, sinon `y_column`)
    plt.ylabel(ylabel if ylabel else y_column)

    # Rotation des étiquettes de l'axe X pour améliorer la lisibilité
    plt.xticks(rotation=rotation, ha='right')

    # Ajouter une grille horizontale si `grid` est activé
    if grid:
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Ajuster automatiquement la mise en page pour éviter le chevauchement des éléments
    plt.tight_layout()

    # Afficher le graphique
    plt.show()

def detect_outliers(data, method="iqr", threshold=1.5, column_name=None):
    """
    Identifie les outliers pour une ou plusieurs colonnes spécifiques ou toutes les variables quantitatives d'un DataFrame.

    Arguments :
    - data : pd.DataFrame : Le DataFrame contenant les données.
    - method : str : La méthode pour identifier les outliers. Peut être "iqr" ou "zscore".
    - threshold : float : Seuil pour identifier les outliers. 
        - Pour "iqr", c'est 1.5 par défaut.
        - Pour "zscore", c'est 3 par défaut.
    - column_name : str, list ou None : Le(s) nom(s) de colonne(s) à analyser. Si None, toutes les colonnes quantitatives sont analysées.
    
    Retourne :
    - Un dictionnaire où les clés sont les noms des colonnes et les valeurs sont les indices des outliers.
    """
    if method not in ["iqr", "zscore"]:
        raise ValueError("La méthode doit être 'iqr' ou 'zscore'.")

    if column_name:
        # Si une seule colonne est donnée, la convertir en liste
        if isinstance(column_name, str):
            column_name = [column_name]
        # Vérifier que les colonnes sont présentes dans le DataFrame
        missing_columns = [col for col in column_name if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Les colonnes suivantes n'existent pas dans le DataFrame : {missing_columns}")
        # Vérifier que toutes les colonnes sont quantitatives
        non_numeric_columns = [col for col in column_name if not pd.api.types.is_numeric_dtype(data[col])]
        if non_numeric_columns:
            raise ValueError(f"Les colonnes suivantes ne sont pas quantitatives : {non_numeric_columns}")
        columns_to_analyze = column_name
    else:
        # Sélectionner toutes les colonnes quantitatives
        columns_to_analyze = data.select_dtypes(include=[np.number]).columns
        if len(columns_to_analyze) == 0:
            raise ValueError("Aucune variable quantitative trouvée dans le DataFrame.")

    outliers = {}

    for column in columns_to_analyze:
        if method == "iqr":
            # Méthode IQR
            q1 = data[column].quantile(0.25)
            q3 = data[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outlier_indices = data[(data[column] < lower_bound) | (data[column] > upper_bound)].index
        elif method == "zscore":
            # Méthode Z-score
            mean = data[column].mean()
            std_dev = data[column].std()
            z_scores = (data[column] - mean) / std_dev
            outlier_indices = data[abs(z_scores) > threshold].index

        outliers[column] = list(outlier_indices)

    return outliers


def replace_values_with_nan(df, replace_dict):
    """
    Remplace certaines valeurs spécifiées dans un dictionnaire par des NaN dans un DataFrame.

    Parameters:
    - df (pd.DataFrame): Le DataFrame à modifier.
    - replace_dict (dict): Un dictionnaire où les clés sont les noms des colonnes 
                           et les valeurs sont des listes de valeurs ou des scalaires
                           à remplacer par NaN.

    Returns:
    - pd.DataFrame: Le DataFrame avec les valeurs remplacées par NaN.
    """
    # Vérifier si le DataFrame est valide
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Le premier paramètre doit être un DataFrame pandas.")
    
    # Vérifier si le dictionnaire est valide
    if not isinstance(replace_dict, dict):
        raise ValueError("Le deuxième paramètre doit être un dictionnaire.")
    
    # Copie du DataFrame pour éviter de modifier l'original
    df = df.copy()

    for column, values_to_replace in replace_dict.items():
        # Vérifier si la colonne existe dans le DataFrame
        if column in df.columns:
            # Si values_to_replace est un scalaire, le convertir en liste
            if not isinstance(values_to_replace, (list, np.ndarray)):
                values_to_replace = [values_to_replace]
            
            # Remplacement direct avec pandas replace (supporte les listes)
            df[column] = df[column].replace(values_to_replace, np.nan)
        else:
            print(f"Avertissement : La colonne '{column}' n'existe pas dans le DataFrame.")
    
    return df

# Liste des essences d'arbres présentes à Paris.
# Cette liste contient les noms communs des différentes espèces d'arbres que l'on peut trouver dans la ville.
# Ces essences peuvent être utilisées pour des analyses sur la biodiversité urbaine, l'entretien ou la gestion des espaces verts.
essences_arbres_paris = ["Abelia", "Abricotier", "Ailante", "Alangium", "Alisier", "Althéa", "Amélanchier", "Arbre de Judée",
    "Arbre aux mouchoirs", "Arbre caramel", "Arbre à perruque", "Argousier", "Aubépine", "Aulne", "Baguenaudier",
    "Bambou", "Bananier", "Bouleau", "Buis", "Cèdre", "Cerisier", "Châtaignier", "Chêne", "Chicot du Canada",
    "Chionanthus", "Chitalpa", "Cognassier", "Cornouiller", "Cotonéaster", "Cyprès", "Daphné", "Éléagnus",
    "Érable", "Faux-poivrier", "Figuier", "Fusain", "Ginkgo", "Grenadier", "Hêtre", "Houx", "If", "Jasmin",
    "Kaki", "Laurier", "Lilas", "Magnolia", "Marronnier", "Mélèze", "Mûrier", "Néflier", "Noisetier", "Noyer",
    "Olivier", "Oranger des Osages", "Orme", "Palmier", "Paulownia", "Pêcher", "Périscope", "Pin", "Platane",
    "Poivrier du Sichuan", "Pommier", "Prunier", "Robinier", "Sapin", "Savonnier", "Sophora", "Sorbier",
    "Sumac", "Tamaris", "Tilleul", "Troène", "Tulipier", "Viorne", "Yuzu", "Zelkova"
]

# Liste des genres botaniques d'arbres présents en France.
# Contrairement à la liste précédente qui contient les noms communs, ici, les arbres sont regroupés par genres botaniques.
# Cette classification est utile pour des analyses scientifiques et botaniques, permettant d'étudier les familles d'arbres 
# et leur répartition dans le territoire français.
genres_arbres_france = [
    "Acer", "Aesculus", "Alnus", "Betula", "Carpinus", "Castanea", "Cedrus", "Celtis", "Cercis", 
    "Corylus", "Cupressus", "Fagus", "Fraxinus", "Ginkgo", "Juglans", "Juniperus", "Larix", 
    "Liquidambar", "Liriodendron", "Magnolia", "Malus", "Morus", "Olea", "Picea", "Pinus", 
    "Platanus", "Populus", "Prunus", "Pseudotsuga", "Pyrus", "Quercus", "Robinia", "Salix", 
    "Sequoiadendron", "Sophora", "Taxodium", "Taxus", "Tilia", "Tsuga", "Ulmus", "Zelkova"
]

especes_arbres_france = [
    # Acer
    "Acer campestre", "Acer platanoides", "Acer pseudoplatanus", "Acer saccharinum", "Acer rubrum",
    # Aesculus
    "Aesculus hippocastanum", "Aesculus x carnea",
    # Alnus
    "Alnus glutinosa", "Alnus incana", "Alnus cordata",
    # Betula
    "Betula pendula", "Betula pubescens",
    # Carpinus
    "Carpinus betulus",
    # Castanea
    "Castanea sativa",
    # Cedrus
    "Cedrus atlantica", "Cedrus libani", "Cedrus deodara",
    # Celtis
    "Celtis australis",
    # Fagus
    "Fagus sylvatica",
    # Fraxinus
    "Fraxinus excelsior", "Fraxinus angustifolia", "Fraxinus ornus",
    # Ginkgo
    "Ginkgo biloba",
    # Juglans
    "Juglans regia", "Juglans nigra",
    # Liquidambar
    "Liquidambar styraciflua",
    # Magnolia
    "Magnolia grandiflora", "Magnolia kobus",
    # Malus
    "Malus sylvestris", "Malus domestica",
    # Morus
    "Morus alba", "Morus nigra",
    # Pinus
    "Pinus nigra", "Pinus sylvestris", "Pinus pinea", "Pinus halepensis", "Pinus pinaster",
    # Platanus
    "Platanus x hispanica", "Platanus orientalis",
    # Populus
    "Populus alba", "Populus nigra", "Populus tremula", "Populus x canadensis",
    # Prunus
    "Prunus avium", "Prunus cerasifera", "Prunus domestica", "Prunus laurocerasus", "Prunus padus",
    # Quercus
    "Quercus robur", "Quercus petraea", "Quercus ilex", "Quercus cerris", "Quercus suber",
    # Robinia
    "Robinia pseudoacacia",
    # Salix
    "Salix alba", "Salix babylonica", "Salix caprea", "Salix fragilis",
    # Tilia
    "Tilia cordata", "Tilia platyphyllos", "Tilia x europaea",
    # Ulmus
    "Ulmus glabra", "Ulmus minor", "Ulmus laevis",
    # Zelkova
    "Zelkova serrata"
]

# Liste des espèces d'arbres présentes en France, classées par genre botanique.
# Cette liste contient les noms scientifiques des espèces d'arbres, ce qui permet une classification plus précise
# et standardisée, notamment pour des analyses scientifiques ou botaniques.
varietes_arbres_france = [
    # Variétés de Platanus (Platanes)
    "x hispanica", "orientalis", "occidentalis",
    
    # Variétés de Tilia (Tilleuls)
    "cordata", "platyphyllos", "x europaea", "henryana", "tomentosa",
    
    # Variétés de Acer (Érables)
    "palmatum", "rubrum", "saccharum", "pseudoplatanus", "campestre", "platanoides",
    
    # Variétés de Quercus (Chênes)
    "robur", "petraea", "cerris", "ilex", "coccifera", "rubra", "suber", "pubescens",
    
    # Variétés de Aesculus (Marronniers)
    "hippocastanum", "carnea", "flava", "indica", "pavia",
    
    # Variétés de Betula (Bouleaux)
    "pendula", "pubescens", "utilis", "nigra",
    
    # Variétés de Fraxinus (Frênes)
    "excelsior", "angustifolia", "ornus", "americana",
    
    # Variétés de Populus (Peupliers)
    "alba", "nigra", "tremula", "deltoides", "balsamifera",
    
    # Variétés de Ulmus (Ormes)
    "minor", "glabra", "laevis", "americana",
    
    # Variétés de Salix (Saules)
    "babylonica", "alba", "fragilis", "purpurea", "caprea",
    
    # Variétés de Pinus (Pins)
    "sylvestris", "nigra", "pinaster", "halepensis", "pinea", "mugo",
    
    # Variétés de Picea (Épicéas)
    "abies", "sitchensis", "engelmannii",
    
    # Variétés de Abies (Sapins)
    "alba", "nordmanniana", "balsamea", "concolor", "pinsapo",
    
    # Variétés de Fagus (Hêtres)
    "sylvatica", "orientalis", "purpurea",
    
    # Variétés de Prunus (Cerisiers, Pruniers, etc.)
    "avium", "cerasifera", "laurocerasus", "padus", "serrulata",
    
    # Variétés de Malus (Pommier)
    "domestica", "sylvestris", "floribunda",
    
    # Variétés de Sorbus (Sorbier)
    "aucuparia", "torminalis", "domestica",
    
    # Variétés de Robinia (Faux acacias)
    "pseudoacacia", "frisia", "umbraculifera",
    
    # Variétés de Ginkgo
    "biloba",
    
    # Variétés de Cedrus (Cèdres)
    "atlantica", "libani", "deodara",
    
    # Variétés de Juglans (Noyers)
    "regia", "nigra",
    
    # Variétés de Carpinus (Charme)
    "betulus",
    
    # Variétés de Alnus (Aulnes)
    "glutinosa", "incana", "viridis",
    
    # Variétés de Liquidambar (Copalme d'Amérique)
    "styraciflua",
    
    # Variétés de Laurus (Lauriers)
    "nobilis",
    
    # Variétés de Taxus (Ifs)
    "baccata", "cuspidata",
    
    # Variétés de Magnolia
    "grandiflora", "stellata", "x soulangeana",
    
    # Variétés de Eucalyptus
    "globulus", "gunnii", "citriodora",
    
    # Variétés de Cupressus (Cyprès)
    "sempervirens", "macrocarpa",
    
    # Variétés de Sequoia
    "sempervirens", "giganteum",
    
    # Variétés de Castanea (Châtaigniers)
    "sativa",
    
    # Variétés de Celtis (Micocouliers)
    "australis", "occidentalis"
]

def summarize_qualitative_variables(data, columns=None):
    """
    Résume les statistiques descriptives de variables qualitatives dans un DataFrame,
    avec des critères de plausibilité spécifiques.

    Arguments :
    - data : pd.DataFrame : Le DataFrame contenant les données.
    - columns : list or str, optional : Liste des colonnes à analyser ou une seule colonne (par défaut analyse toutes les colonnes qualitatives).

    Retourne :
    - Un dictionnaire où chaque clé est le nom d'une colonne qualitative,
      et la valeur est un résumé des statistiques descriptives, incluant la plausibilité.
    """
    # Filtrer les colonnes qualitatives
    all_qualitative_columns = data.select_dtypes(include=['object', 'category']).columns

    # Déterminer les colonnes à analyser
    if columns is None:
        qualitative_columns = all_qualitative_columns
    else:
        if isinstance(columns, str):  # Si une seule colonne est donnée
            columns = [columns]
        qualitative_columns = [col for col in columns if col in all_qualitative_columns]
        if len(qualitative_columns) == 0:
            raise ValueError("Aucune colonne qualitative valide spécifiée dans les paramètres.")

    # Critères de plausibilité pour chaque variable
    plausibility_criteria = {
        "type_emplacement": ["Arbre"],
        "domanialite": ["Alignement", "Jardin", "CIMETIERE", "DASCO", "PERIPHERIQUE", "DJS", "DFPE", "DAC", "DASES"],
        "arrondissement": [str(i) + "e" for i in range(1, 21)],  # 1er à 20e
        "complement_addresse": None,  # Pas de critère spécifique, car peut varier largement
        "lieu": None,  # Peut être libre
        "id_emplacement": None,  # Doit être unique (plausibilité vérifiée ailleurs)
        "libelle_francais": lambda x: x in essences_arbres_paris,  # Exemple de catégories possibles
        "genre": lambda x: x in genres_arbres_france,  # Genres connus
        "espece": lambda x: x in especes_arbres_france,  # Texte descriptif
        "variete": lambda x: x in varietes_arbres_france, 
        "stade_developpement": lambda x: x in ["A", "JA", "J", "M"]
    }

    summary = {}

    for column in qualitative_columns:
        # Calcul des statistiques pour chaque colonne qualitative
        counts = data[column].value_counts()  # Fréquences absolues
        mode = data[column].mode()  # Mode (valeur(s) la plus fréquente)
        unique_values = data[column].unique().tolist()

        # Vérification de la plausibilité
        criteria = plausibility_criteria.get(column)
        if criteria is not None:
            invalid_values = [val for val in unique_values if val not in criteria]
        else:
            invalid_values = []  # Pas de critères définis

        summary[column] = {
            "Fréquences absolues": counts.to_dict(),
            "Mode(s)": mode.tolist(),
            "Nombre de catégories uniques": data[column].nunique(),
            "Nombre de valeurs manquantes": data[column].isnull().sum(),
            "Critères de plausibilité": criteria,
            "Valeurs invalides": invalid_values
        }

    return summary


def evaluate_outliers_with_criteria(data, outliers_dict=None, show_only_aberrant=False):
    """
    Évalue la plausibilité des outliers pour chaque variable d'un DataFrame 
    en fonction de critères spécifiques. Optionnellement, affiche uniquement les valeurs aberrantes.
    
    Arguments :
    - data : pd.DataFrame : Le DataFrame contenant les données.
    - outliers_dict : dict (optionnel) : Un dictionnaire où les clés sont les noms des variables 
                             et les valeurs sont les indices ou les valeurs des outliers.
    - show_only_aberrant : bool : Si True, affiche uniquement les valeurs jugées aberrantes.
                             
    Retourne :
    - dict : Un dictionnaire indiquant si chaque outlier est plausible ou aberrant.
    """

    # Définir les critères de plausibilité pour chaque variable
    plausibility_criteria = {
        "id": lambda x: isinstance(x, int) and x > 0,  # L'identifiant doit être un entier positif
        "type_emplacement": lambda x: x in ["Arbre"],  # Catégories connues
        "domanialite": lambda x: x in ["Alignement", "Jardin", "CIMETIERE", "DASCO", "PERIPHERIQUE", "DJS", "DFPE", "DAC", "DASES"],  # Dom
        "arrondissement": lambda x: x.isdigit() and 1 <= int(x) <= 20,  # Entre 1 et 20
        "complement_addresse": lambda x: isinstance(x, str) and len(x) < 100,  # Texte court
        "numero": lambda x: isinstance(x, int) and x > 0,  # Numéro d'adresse positif
        "lieu": lambda x: isinstance(x, str) and len(x) < 100,  # Texte court
        "id_emplacement": lambda x: isinstance(x, str) and len(x) < 50,  # Identifiant court
        "libelle_francais": lambda x: x in essences_arbres_paris,  # Texte court
        "genre": lambda x: x in genres_arbres_france,  # Genres connus
        "espece": lambda x: x in especes_arbres_france,  # Texte descriptif
        "variete": lambda x: x in varietes_arbres_france,  # Texte descriptif
        "circonference_cm": lambda x: 0 <= x <= 350,  # Entre 0 et 350 cm
        "hauteur_m": lambda x: 0 <= x <= 40,  # Entre 1 et 115,92 mètres
        "stade_developpement": lambda x: x in ["A", "JA", "J", "M"],  # Stades connus
        "remarquable": lambda x: x in [True, False],  # Booléen
        "geo_point_2d_a": lambda x: 48.81 <= x <= 48.90,  # Latitude de Paris
        "geo_point_2d_b": lambda x: 2.22 <= x <= 2.46,  # Longitude de Paris
    }

    # Résultat final
    results = {}

    if outliers_dict is not None:
        # Si un dictionnaire de outliers est fourni
        for variable, outlier_indices in outliers_dict.items():
            if variable not in plausibility_criteria:
                results[variable] = "Aucun critère défini"
                continue

            # Récupérer les valeurs des outliers pour la variable
            outlier_values = data.loc[outlier_indices, variable]

            # Vérifier chaque outlier par rapport au critère
            evaluation = {
                value: "Aberrant" if not plausibility_criteria[variable](value) else "Plausible"
                for value in outlier_values
            }

            # Filtrer les valeurs aberrantes si demandé
            if show_only_aberrant:
                evaluation = {k: v for k, v in evaluation.items() if v == "Aberrant"}

            results[variable] = evaluation

    return results

def filter_rows_by_value(df, column_name):
    """
    Filtre les lignes d'un DataFrame où une colonne donnée a pour valeur 1 ou True.

    Args:
        df (pd.DataFrame): Le DataFrame à filtrer.
        column_name (str): Le nom de la colonne à vérifier.

    Returns:
        pd.DataFrame: Un DataFrame contenant uniquement les lignes où la valeur est 1 ou True.
    """
    if column_name not in df.columns:
        raise ValueError(f"La colonne '{column_name}' n'existe pas dans le DataFrame.")
    
    # Filtrer les lignes où la valeur est 1 ou True
    filtered_df = df[df[column_name] == 1]
    return filtered_df

def check_unique_columns(dataframe, columns=None):
    """
    Vérifie si une ou plusieurs colonnes d'un DataFrame contiennent des valeurs uniques.

    Parameters:
        dataframe (pd.DataFrame): Le DataFrame à analyser.
        columns (list or str, optional): Liste des colonnes à vérifier ou une seule colonne.
                                         Si None, toutes les colonnes du DataFrame sont analysées.

    Returns:
        dict: Un dictionnaire indiquant pour chaque colonne si elle contient des valeurs uniques.
              Clé : nom de la colonne, Valeur : True (unique) ou False (non unique).
    """
    # Si aucune colonne n'est spécifiée, vérifier toutes les colonnes
    if columns is None:
        columns = dataframe.columns
    elif isinstance(columns, str):
        columns = [columns]
    
    # Vérifier les colonnes spécifiées
    uniqueness = {}
    for column in columns:
        if column in dataframe.columns:
            uniqueness[column] = dataframe[column].is_unique
        else:
            raise ValueError(f"La colonne '{column}' n'existe pas dans le DataFrame.")
    
    return uniqueness