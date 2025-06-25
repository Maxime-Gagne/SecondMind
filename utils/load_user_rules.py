import importlib.util
import os
import yaml

def get_active_user():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'rag_config.yaml.txt')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('user', 'default')

def load_user_rules():
    user = get_active_user()
    rules_path = os.path.join(os.path.dirname(__file__), '..', 'user_rules', f'{user}_rules.py')

    if not os.path.isfile(rules_path):
        print(f"[!] Aucun fichier de règles trouvé pour l'utilisateur : {user}")
        return {}

    spec = importlib.util.spec_from_file_location("user_rules", rules_path)
    user_rules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user_rules)

    if hasattr(user_rules, 'rules'):
        print(f"[✓] Règles utilisateur chargées pour : {user}")
        return user_rules.rules
    else:
        print(f"[!] Le fichier {rules_path} ne contient pas de variable 'rules'.")
        return {}

# Exemple d'utilisation autonome
if __name__ == "__main__":
    rules = load_user_rules()
    print(rules)
