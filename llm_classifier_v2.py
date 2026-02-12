"""
llm_classifier.py
Module pour classifier les contrats Solidity avec un LLM (flexible 4 ou 8 labels)
"""
import requests

MODEL_LLM = "llama3.2"
BASE_URL = "http://localhost:11434/api/generate"

# Configurations des labels par dataset
LABELS_CONFIG = {
    4: {
        0: "Dangerous delegatecall",
        1: "Integer overflow",
        2: "Reentrancy",
        3: "Timestamp dependency",
        4: "Normal"
    },
    8: {
        0: "Block number dependency (BN)",
        1: "Dangerous delegatecall (DE)",
        2: "Ether frozen (EF)",
        3: "Ether strict equality (SE)",
        4: "Integer overflow (OF)",
        5: "Reentrancy (RE)",
        6: "Timestamp dependency (TP)",
        7: "Unchecked external call (UC)",
        8: "Normal"
    }
}


def creer_prompt_classification(code_solidity, nb_labels=8):
    """
    Crée le prompt pour classifier un contrat Solidity.

    Args:
        code_solidity: Code du contrat à analyser
        nb_labels: 4 ou 8 selon le dataset utilisé
    """
    labels_dict = LABELS_CONFIG.get(nb_labels)

    if labels_dict is None:
        raise ValueError(f"Configuration non supportée pour {nb_labels} labels. Utilisez 4 ou 8.")

    # Construire la liste des options
    max_label = len(labels_dict) - 1
    options = "\n".join([f"- {i} si le contrat a une vulnérabilité de type \"{desc}\""
                         if i < max_label else f"- {i} si le contrat est normal (sans vulnérabilité)"
                         for i, desc in labels_dict.items()])

    prompt = f"""Analyse ce contrat Solidity et identifie s'il contient une vulnérabilité.

Réponds UNIQUEMENT avec UN SEUL chiffre entre 0 et {max_label} :
{options}

IMPORTANT : Réponds UNIQUEMENT avec le chiffre, rien d'autre.

Contrat Solidity à analyser :
{code_solidity}

Réponse (un seul chiffre) :"""

    return prompt


def ask_ollama(prompt, model=MODEL_LLM):
    """
    Envoie un prompt à Ollama et retourne la réponse.
    """
    try:
        response = requests.post(
            BASE_URL,
            json={
                'model': model,
                'prompt': prompt,
                'stream': False
            },
            timeout=60
        )

        if response.status_code == 200:
            return response.json().get('response', '').strip()
        else:
            return f"Erreur: {response.status_code}"
    except Exception as e:
        return f"Erreur: {str(e)}"


def classifier_contrat(code_solidity, model=MODEL_LLM, nb_labels=8):
    """
    Classifie un contrat Solidity et retourne le label.

    Args:
        code_solidity: Code du contrat
        model: Modèle LLM à utiliser
        nb_labels: 4 ou 8 selon le dataset
    """
    prompt = creer_prompt_classification(code_solidity, nb_labels)
    reponse = ask_ollama(prompt, model)

    # Déterminer les chiffres valides selon le nombre de labels
    max_label = len(LABELS_CONFIG[nb_labels]) - 1
    valid_chars = ''.join(str(i) for i in range(max_label + 1))

    # Extraire le premier chiffre valide de la réponse
    for char in reponse:
        if char.isdigit() and char in valid_chars:
            return int(char)

    # Si aucun chiffre valide trouvé, retourner -1 (erreur)
    return -1


def get_labels_dict(nb_labels):
    """
    Retourne le dictionnaire de labels pour le nombre spécifié.
    """
    return LABELS_CONFIG.get(nb_labels, {})


# Test du module
if __name__ == "__main__":
    # Exemple de code Solidity simple
    code_test = """
    pragma solidity ^0.4.15;

    contract SimpleContract {
        uint256 public value;

        function setValue(uint256 _value) public {
            value = _value;
        }
    }
    """

    print("=== TEST AVEC 4 LABELS ===")
    resultat_4 = classifier_contrat(code_test, nb_labels=4)
    print(f"Résultat : {resultat_4}")
    print(f"Signification : {LABELS_CONFIG[4].get(resultat_4, 'Erreur')}")

    print("\n=== TEST AVEC 8 LABELS ===")
    resultat_8 = classifier_contrat(code_test, nb_labels=8)
    print(f"Résultat : {resultat_8}")
    print(f"Signification : {LABELS_CONFIG[8].get(resultat_8, 'Erreur')}")