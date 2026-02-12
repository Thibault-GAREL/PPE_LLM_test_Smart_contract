"""
llm_classifier.py
Module pour classifier les contrats Solidity avec un LLM
"""
import requests

MODEL_LLM = "llama3.2"
BASE_URL = "http://localhost:11434/api/generate"


def creer_prompt_classification(code_solidity):
    """
    Crée le prompt pour classifier un contrat Solidity.
    """
    prompt = f"""Analyse ce contrat Solidity et identifie s'il contient une vulnérabilité.

Réponds UNIQUEMENT avec UN SEUL chiffre entre 0 et 4 :
- 0 si le contrat a une vulnérabilité de type "dangerous delegatecall"
- 1 si le contrat a une vulnérabilité de type "integer overflow"
- 2 si le contrat a une vulnérabilité de type "reentrancy"
- 3 si le contrat a une vulnérabilité de type "timestamp dependency"
- 4 si le contrat est normal (sans vulnérabilité)

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


def classifier_contrat(code_solidity, model=MODEL_LLM):
    """
    Classifie un contrat Solidity et retourne le label (0-4).
    """
    prompt = creer_prompt_classification(code_solidity)
    reponse = ask_ollama(prompt, model)

    # Extraire le premier chiffre de la réponse
    for char in reponse:
        if char.isdigit() and char in '01234':
            return int(char)

    # Si aucun chiffre valide trouvé, retourner -1 (erreur)
    return reponse


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

    print("Test de classification...")
    print(f"Code à analyser : {code_test[:100]}...")
    print("\nClassification en cours...")

    resultat = classifier_contrat(code_test)
    print(f"\nRésultat : {resultat}")

    labels = {
        0: "Dangerous delegatecall",
        1: "Integer overflow",
        2: "Reentrancy",
        3: "Timestamp dependency",
        4: "Normal",
        -1: "Erreur de classification"
    }
    print(f"Signification : {labels.get(resultat, 'Inconnu')}")