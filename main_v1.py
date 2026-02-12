"""
main_v1.py
Programme principal pour classifier les contrats Solidity
"""
import pandas as pd
from csv_loader import lire_csv, afficher_info_dataset
from llm_classifier import classifier_contrat
from tqdm import tqdm


def classifier_dataset(df, model="mistral", limite=None):
    """
    Classifie tous les contrats du dataset avec le LLM.

    Args:
        df: DataFrame contenant les contrats
        model: Modèle LLM à utiliser
        limite: Nombre maximum de contrats à classifier (None = tous)

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
    print("Cela peut prendre du temps...\n")

    # Classifier chaque contrat avec barre de progression
    for idx in tqdm(range(nb_lignes), desc="Classification"):
        code = df_result.iloc[idx]['code']
        prediction = classifier_contrat(code, model=model)
        df_result.at[idx, 'llm_prediction'] = prediction

    return df_result


def calculer_precision(df):
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

    # Matrice de confusion simplifiée
    print("\n=== DISTRIBUTION DES PRÉDICTIONS ===")
    labels_names = {
        0: "Dangerous delegatecall",
        1: "Integer overflow",
        2: "Reentrancy",
        3: "Timestamp dependency",
        4: "Normal"
    }

    print("\nLabel réel vs Prédiction:")
    for label in sorted(df_valide['label_encoded'].unique()):
        count = len(df_valide[df_valide['label_encoded'] == label])
        print(f"\n{int(label)} - {labels_names.get(int(label), 'Inconnu')} ({count} contrats):")
        predictions = df_valide[df_valide['label_encoded'] == label]['llm_prediction'].value_counts()
        for pred, cnt in predictions.items():
            print(f"  -> Prédit comme {int(pred)}: {cnt} fois ({cnt/count*100:.1f}%)")


def sauvegarder_resultats(df, fichier_sortie="resultats_classification.csv"):
    """
    Sauvegarde les résultats dans un nouveau CSV.
    """
    df.to_csv(fichier_sortie, index=False)
    print(f"\n✓ Résultats sauvegardés dans : {fichier_sortie}")


# Programme principal
if __name__ == "__main__":
    # 1. Charger les données
    print("=== CHARGEMENT DES DONNÉES ===")
    fichier = "archive/SC_4label.csv"
    # fichier = "archive/SC_Vuln_8label.csv"
    # nb_label = 8
    df = lire_csv(fichier)
    afficher_info_dataset(df)

    # 2. Classifier les contrats
    # ATTENTION : Commencez avec un petit nombre pour tester !
    LIMITE = 50  # Modifier ce nombre (None = tous les contrats)
    MODEL = "qwen2.5-coder:7b"  # Changer le modèle si besoin   llama3.2

    df_resultat = classifier_dataset(df, model=MODEL, limite=LIMITE)

    # 3. Calculer la précision
    calculer_precision(df_resultat)

    # 4. Sauvegarder les résultats
    sauvegarder_resultats(df_resultat)

    # 5. Afficher quelques exemples
    print("\n=== EXEMPLES DE PRÉDICTIONS ===")
    for i in range(min(5, len(df_resultat))):
        row = df_resultat.iloc[i]
        print(f"\nContrat {i+1} ({row['filename']}):")
        print(f"  Label réel : {row['label_encoded']}")
        print(f"  Prédiction LLM : {row['llm_prediction']}")
        print(f"  Correct : {'✓' if row['label_encoded'] == str(row['llm_prediction']) else '✗'}")