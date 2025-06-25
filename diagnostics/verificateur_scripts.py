import re
import sys
from pathlib import Path

# Règles critiques
SCRIPTS_VALIDES = {
    "vectorize_chatgpt_Online.py",
    "vectorize_LOCAL.py",
    "app_gradio_faiss_ONLINE.py",
    "fix_faiss_index.py",
    "log_server.py",
    "diagnostic_faiss.py",
    "diagnostic_faiss_fixed.py",
    "debug_main_fix.py"
}

# Signatures interdites (regex plus souples)
IMPORTS_INTERDITS = [
    re.compile(r"from\s+langchain\.vectorstores\.faiss\s+import\s+dependable_faiss_import"),
    re.compile(r"os\.remove\s*\("),
]

def verifier_py(script_path: Path, mode_relax=False):
    erreurs = []
    avertissements = []

    if not script_path.exists():
        erreurs.append(("?", f"❌ Fichier introuvable : {script_path.name}"))
        return erreurs, avertissements

    if script_path.name not in SCRIPTS_VALIDES:
        message = f"⚠️ Script non listé comme autorisé : {script_path.name}"
        if mode_relax:
            avertissements.append(("?", message))
        else:
            erreurs.append(("?", f"❌ {message}"))

    try:
        with open(script_path, encoding="utf-8") as f:
            lignes = f.readlines()
    except Exception as e:
        erreurs.append(("?", f"❌ Erreur de lecture : {e}"))
        return erreurs, avertissements

    for i, ligne in enumerate(lignes, 1):
        for rex in IMPORTS_INTERDITS:
            if rex.search(ligne):
                erreurs.append((i, f"❌ Pattern interdit détecté : `{rex.pattern}`"))

    return erreurs, avertissements

def analyser_dossier(dossier: Path, mode_relax=False):
    fichiers = sorted(dossier.glob("*.py"))
    total, fautifs = 0, 0
    for fichier in fichiers:
        total += 1
        erreurs, avertissements = verifier_py(fichier, mode_relax)
        if erreurs or avertissements:
            print(f"\n📄 {fichier.name}")
            for ligne, msg in erreurs:
                print(f"  Ligne {ligne} - {msg}")
            for ligne, msg in avertissements:
                print(f"  Ligne {ligne} - {msg}")
            if erreurs:
                fautifs += 1

    print("\n===== Résultat global =====")
    if fautifs == 0:
        print(f"✅ {total} script(s) analysé(s). Aucun problème critique détecté.")
    else:
        print(f"❌ {fautifs} script(s) non conformes sur {total}.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python verificateur_scripts.py chemin\\vers\\dossier [--relax]")
        sys.exit(1)

    dossier = Path(sys.argv[1])
    relax_mode = "--relax" in sys.argv
    if not dossier.exists() or not dossier.is_dir():
        print("❌ Chemin invalide ou non accessible.")
        sys.exit(1)

    analyser_dossier(dossier, relax_mode)
