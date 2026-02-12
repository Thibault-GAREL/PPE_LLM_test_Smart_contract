"""
main.py
Programme principal pour classifier les contrats Solidity (flexible 4 ou 8 labels)
"""
import pandas as pd
from csv_loader import lire_csv, afficher_info_dataset
from llm_classifier_v2 import classifier_contrat, get_labels_dict
from tqdm import tqdm

# ========================================
# CONFIGURATION - MODIFIEZ ICI
# ========================================
CONFIG = {
    # Choisir le dataset : "4labels" ou "8labels"
    "dataset": "8labels",  # <-- CHANGEZ ICI

    # Nombre de contrats à classifier (None = tous)
    "limite": 100,  # <-- CHANGEZ ICI

    # Modèle LLM à utiliser
    "model": "llama3.2",  # <-- CHANGEZ ICI :  llama3.2 / kimi-k2.5:cloud / qwen2.5-coder:7b / qwen3-coder:30b / deepseek-coder:6.7b
}

# Mapping des fichiers selon le dataset choisi
DATASETS = {
    "4labels": {
        "fichier": "archive/SC_4label.csv",
        "nb_labels": 4,
        "fichier_sortie": "resultats_classification_4labels.csv"
    },
    "8labels": {
        "fichier": "archive/SC_Vuln_8label.csv",
        "nb_labels": 8,
        "fichier_sortie": "resultats_classification_8labels.csv"
    }
}


# ========================================


def classifier_dataset(df, model="mistral", limite=None, nb_labels=8):
    """
    Classifie tous les contrats du dataset avec le LLM.

    Args:
        df: DataFrame contenant les contrats
        model: Modèle LLM à utiliser
        limite: Nombre maximum de contrats à classifier (None = tous)
        nb_labels: Nombre de labels (4 ou 8)

    Returns:
        DataFrame avec la nouvelle colonne 'llm_prediction'
    """
    # Créer une copie du DataFrame
    df_result = df.copy()

    # Ajouter une colonne pour les prédictions
    df_result['llm_prediction'] = -1

    # Limiter le nombre de lignes si demandé
    nb_lignes = len(df_result) if limite is None else min(limite, len(df_result))

    print(f"\nClassification de {nb_lignes} contrats avec le modèle {model}...")
    print(f"Dataset : {nb_labels} labels")
    print("Cela peut prendre du temps...\n")

    # Classifier chaque contrat avec barre de progression
    for idx in tqdm(range(nb_lignes), desc="Classification"):
        code = df_result.iloc[idx]['code']
        prediction = classifier_contrat(code, model=model, nb_labels=nb_labels)
        df_result.at[idx, 'llm_prediction'] = prediction

    return df_result


def calculer_precision(df, nb_labels=8):
    """
    Calcule la précision de la classification.
    """
    # Convertir les colonnes en numérique
    df['label_encoded'] = pd.to_numeric(df['label_encoded'], errors='coerce')
    df['llm_prediction'] = pd.to_numeric(df['llm_prediction'], errors='coerce')

    # Filtrer les prédictions valides (-1 = erreur)
    df_valide = df[df['llm_prediction'] != -1]

    if len(df_valide) == 0:
        print("Aucune prédiction valide!")
        return

    # Calculer la précision
    correct = (df_valide['label_encoded'] == df_valide['llm_prediction']).sum()
    total = len(df_valide)
    precision = (correct / total) * 100

    print(f"\n=== RÉSULTATS ===")
    print(f"Prédictions valides : {total}")
    print(f"Prédictions correctes : {correct}")
    print(f"Précision : {precision:.2f}%")

    # Récupérer les noms des labels
    labels_names = get_labels_dict(nb_labels)

    # Matrice de confusion simplifiée
    print("\n=== DISTRIBUTION DES PRÉDICTIONS ===")
    print("\nLabel réel vs Prédiction:")
    for label in sorted(df_valide['label_encoded'].unique()):
        count = len(df_valide[df_valide['label_encoded'] == label])
        label_name = labels_names.get(int(label), 'Inconnu')
        print(f"\n{int(label)} - {label_name} ({count} contrats):")
        predictions = df_valide[df_valide['label_encoded'] == label]['llm_prediction'].value_counts()
        for pred, cnt in predictions.items():
            pred_name = labels_names.get(int(pred), 'Inconnu')
            print(f"  -> Prédit comme {int(pred)} ({pred_name}): {cnt} fois ({cnt / count * 100:.1f}%)")


def afficher_matrice_confusion(df, nb_labels=8):
    """
    Affiche une matrice de confusion détaillée.
    """
    try:
        from sklearn.metrics import confusion_matrix, classification_report
    except ImportError:
        print("\n⚠ sklearn non installé, matrice de confusion détaillée non disponible")
        print("Installez avec: pip install scikit-learn")
        return

    # Filtrer les prédictions valides
    df_valide = df[df['llm_prediction'] != -1].copy()

    if len(df_valide) == 0:
        print("Aucune prédiction valide pour la matrice de confusion!")
        return

    y_true = df_valide['label_encoded'].astype(int)
    y_pred = df_valide['llm_prediction'].astype(int)

    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)

    max_label = len(get_labels_dict(nb_labels)) - 1

    print("\n=== MATRICE DE CONFUSION ===")
    print("Lignes = Labels réels, Colonnes = Prédictions")
    print("\n     ", end="")
    for i in range(max_label + 1):
        print(f"{i:4}", end="")
    print()

    for i, row in enumerate(cm):
        print(f"{i:2} | ", end="")
        for val in row:
            print(f"{val:4}", end="")
        print()

    # Rapport de classification
    labels_dict = get_labels_dict(nb_labels)

    # Créer des noms courts pour le rapport
    if nb_labels == 4:
        labels_names = ["Delegatecall", "Overflow", "Reentrancy", "Timestamp", "Normal"]
    else:
        labels_names = ["BN", "DE", "EF", "SE", "OF", "RE", "TP", "UC", "Normal"]

    print("\n=== RAPPORT DE CLASSIFICATION ===")
    print(classification_report(y_true, y_pred, target_names=labels_names, zero_division=0))


def sauvegarder_resultats(df, fichier_sortie):
    """
    Sauvegarde les résultats dans un nouveau CSV.
    """
    df.to_csv(fichier_sortie, index=False)
    print(f"\n✓ Résultats sauvegardés dans : {fichier_sortie}")


def afficher_exemples(df_resultat, nb_labels=8):
    """
    Affiche quelques exemples de prédictions.
    """
    print("\n=== EXEMPLES DE PRÉDICTIONS ===")
    labels_dict = get_labels_dict(nb_labels)

    # Créer des noms courts
    if nb_labels == 4:
        labels_short = {0: "Delegatecall", 1: "Overflow", 2: "Reentrancy",
                        3: "Timestamp", 4: "Normal", -1: "Erreur"}
    else:
        labels_short = {0: "BN", 1: "DE", 2: "EF", 3: "SE", 4: "OF",
                        5: "RE", 6: "TP", 7: "UC", 8: "Normal", -1: "Erreur"}

    for i in range(min(5, len(df_resultat))):
        row = df_resultat.iloc[i]
        label_reel = int(row['label_encoded']) if pd.notna(row['label_encoded']) else -1
        label_pred = int(row['llm_prediction']) if pd.notna(row['llm_prediction']) else -1

        print(f"\nContrat {i + 1} ({row.get('filename', 'N/A')}):")
        print(f"  Label réel : {label_reel} ({labels_short.get(label_reel, 'Erreur')})")
        print(f"  Prédiction LLM : {label_pred} ({labels_short.get(label_pred, 'Erreur')})")
        print(f"  Correct : {'✓' if label_reel == label_pred else '✗'}")


# Programme principal
if __name__ == "__main__":
    # Vérifier la configuration
    dataset_choisi = CONFIG["dataset"]

    if dataset_choisi not in DATASETS:
        print(f"❌ Erreur: Dataset '{dataset_choisi}' non reconnu.")
        print(f"Choisissez parmi : {list(DATASETS.keys())}")
        exit(1)

    dataset_config = DATASETS[dataset_choisi]

    print("=" * 60)
    print(f"CONFIGURATION ACTIVE : {dataset_choisi.upper()}")
    print("=" * 60)
    print(f"Fichier : {dataset_config['fichier']}")
    print(f"Nombre de labels : {dataset_config['nb_labels']}")
    print(f"Modèle LLM : {CONFIG['model']}")
    print(f"Limite : {CONFIG['limite'] if CONFIG['limite'] else 'Tous les contrats'}")
    print("=" * 60)

    # 1. Charger les données
    print("\n=== CHARGEMENT DES DONNÉES ===")
    df = lire_csv(dataset_config['fichier'])
    afficher_info_dataset(df)

    # 2. Classifier les contrats
    df_resultat = classifier_dataset(
        df,
        model=CONFIG['model'],
        limite=CONFIG['limite'],
        nb_labels=dataset_config['nb_labels']
    )

    # 3. Calculer la précision
    calculer_precision(df_resultat, nb_labels=dataset_config['nb_labels'])

    # 4. Afficher la matrice de confusion
    afficher_matrice_confusion(df_resultat, nb_labels=dataset_config['nb_labels'])

    # 5. Sauvegarder les résultats
    sauvegarder_resultats(df_resultat, dataset_config['fichier_sortie'])

    # 6. Afficher quelques exemples
    afficher_exemples(df_resultat, nb_labels=dataset_config['nb_labels'])