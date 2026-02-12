"""
csv_loader.py
Module pour charger les données CSV de contrats Solidity
"""
import pandas as pd
import csv


def lire_csv(fichier_path):
    """
    Lit un CSV contenant du code avec des délimiteurs.
    Utilise le module csv natif pour une lecture robuste.
    """
    with open(fichier_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(
            f,
            delimiter=',',
            quotechar='"',
            doublequote=True,
            skipinitialspace=True
        )
        data = list(reader)

    # Convertir en DataFrame pandas
    df = pd.DataFrame(data)

    # Renommer la première colonne vide si elle existe
    if '' in df.columns:
        df = df.rename(columns={'': 'index'})

    return df


def afficher_info_dataset(df):
    """
    Affiche les informations du dataset.
    """
    print(f"Nombre de lignes : {len(df)}")
    print(f"Colonnes : {df.columns.tolist()}")

    # Distribution des labels
    if 'label_encoded' in df.columns:
        print("\nDistribution des labels :")
        print("  0 = Dangerous delegatecall")
        print("  1 = Integer overflow")
        print("  2 = Reentrancy")
        print("  3 = Timestamp dependency")
        print("  4 = Normal")
        print("\nCompte par label :")
        print(df['label_encoded'].value_counts().sort_index())


# Test du module
if __name__ == "__main__":
    fichier = "archive/SC_4label.csv"
    df = lire_csv(fichier)
    afficher_info_dataset(df)

    print("\n=== PREMIÈRE LIGNE ===\n")
    premiere_ligne = df.iloc[0]

    for colonne, valeur in premiere_ligne.items():
        print(f"{colonne}:")
        if colonne == 'code' and len(str(valeur)) > 200:
            print(f"  {valeur[:200]}...")
            print(f"  [Code de {len(valeur)} caractères au total]")
        else:
            print(f"  {valeur}")
        print()