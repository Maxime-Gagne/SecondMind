# -*- coding: utf-8 -*-
"""
Script de réparation et vérification des index FAISS
Version corrigée avec chemins absolus et améliorations
"""

import os
import sys
import faiss
import pickle
import numpy as np
from datetime import datetime
import logging
from sentence_transformers import SentenceTransformer

# Configuration des chemins absolus
BASE_DIR = r"C:\Users\rag_personnel\Logs"
INDEX_DIR = os.path.join(BASE_DIR, "vector_index_chatgpt")
INDEX_FILE = os.path.join(INDEX_DIR, "index.pkl")
FAISS_INDEX = os.path.join(INDEX_DIR, "index.faiss")
CONVERSATIONS_FILE = os.path.join(BASE_DIR, "conversations_extraites.txt")
LOG_FILE = os.path.join(BASE_DIR, "fix_faiss_index.log")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def check_files_status():
    """Vérifie l'état des fichiers nécessaires"""
    print("🔍 VÉRIFICATION DES FICHIERS")
    print("=" * 50)
    
    files_to_check = [
        ("Conversations source", CONVERSATIONS_FILE),
        ("Index PKL", INDEX_FILE),
        ("Index FAISS", FAISS_INDEX),
        ("Dossier index", INDEX_DIR)
    ]
    
    status = {}
    for name, path in files_to_check:
        exists = os.path.exists(path)
        size = os.path.getsize(path) / (1024*1024) if exists and os.path.isfile(path) else 0
        
        status[name] = {
            'exists': exists,
            'path': path,
            'size_mb': size
        }
        
        status_icon = "✅" if exists else "❌"
        size_info = f"({size:.1f} MB)" if exists and os.path.isfile(path) else ""
        print(f"{status_icon} {name}: {exists} {size_info}")
        print(f"   📁 {path}")
    
    return status

def load_and_verify_data():
    """Charge et vérifie les données existantes"""
    print("\n📊 ANALYSE DES DONNÉES")
    print("=" * 50)
    
    try:
        # Vérification du fichier PKL
        if os.path.exists(INDEX_FILE):
            print("📥 Chargement de l'index PKL...")
            with open(INDEX_FILE, 'rb') as f:
                data = pickle.load(f)
            
            texts = data.get('texts', [])
            embeddings = data.get('embeddings', [])
            
            print(f"✅ Texts chargés: {len(texts)}")
            print(f"✅ Embeddings chargés: {len(embeddings) if embeddings is not None else 0}")
            
            if len(embeddings) > 0:
                print(f"📐 Dimension des embeddings: {np.array(embeddings).shape}")
            
            return texts, embeddings
        else:
            print("❌ Fichier PKL introuvable")
            return None, None
            
    except Exception as e:
        print(f"❌ Erreur lors du chargement PKL: {e}")
        logging.error(f"Erreur chargement PKL: {e}")
        return None, None

def verify_faiss_index(embeddings):
    """Vérifie et répare l'index FAISS"""
    print("\n🔧 VÉRIFICATION INDEX FAISS")
    print("=" * 50)
    
    try:
        if os.path.exists(FAISS_INDEX):
            print("📥 Chargement de l'index FAISS existant...")
            index = faiss.read_index(FAISS_INDEX)
            
            print(f"✅ Index FAISS chargé")
            print(f"📊 Nombre de vecteurs: {index.ntotal}")
            print(f"📐 Dimension: {index.d}")
            
            # Test de recherche
            if embeddings is not None and len(embeddings) > 0:
                test_query = np.array([embeddings[0]]).astype('float32')
                distances, indices = index.search(test_query, 1)
                print(f"✅ Test de recherche réussi")
                
            return True, index
        else:
            print("❌ Index FAISS introuvable")
            return False, None
            
    except Exception as e:
        print(f"❌ Erreur avec l'index FAISS: {e}")
        logging.error(f"Erreur FAISS: {e}")
        return False, None

def rebuild_faiss_index(texts, embeddings):
    """Reconstruit l'index FAISS à partir des embeddings"""
    print("\n🔨 RECONSTRUCTION DE L'INDEX FAISS")
    print("=" * 50)
    
    try:
        if embeddings is None or len(embeddings) == 0:
            print("❌ Pas d'embeddings disponibles pour la reconstruction")
            return False
        
        # Conversion en numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        
        print(f"📐 Dimension des embeddings: {dimension}")
        print(f"📊 Nombre d'embeddings: {len(embeddings_array)}")
        
        # Création de l'index FAISS
        print("🏗️ Création de l'index FAISS...")
        index = faiss.IndexFlatL2(dimension)
        
        # Ajout des vecteurs
        print("➕ Ajout des vecteurs à l'index...")
        index.add(embeddings_array)
        
        # Sauvegarde
        print("💾 Sauvegarde de l'index FAISS...")
        faiss.write_index(index, FAISS_INDEX)
        
        print(f"✅ Index FAISS reconstruit avec succès!")
        print(f"📊 {index.ntotal} vecteurs indexés")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la reconstruction: {e}")
        logging.error(f"Erreur reconstruction FAISS: {e}")
        return False

def regenerate_from_source():
    """Régénère tout depuis le fichier source"""
    print("\n🔄 RÉGÉNÉRATION COMPLÈTE")
    print("=" * 50)
    
    try:
        if not os.path.exists(CONVERSATIONS_FILE):
            print(f"❌ Fichier source introuvable: {CONVERSATIONS_FILE}")
            return False
        
        # Chargement du modèle
        print("🤖 Chargement du modèle SentenceTransformer...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Lecture des conversations
        print("📖 Lecture des conversations...")
        with open(CONVERSATIONS_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Découpage en chunks
        chunks = []
        current_chunk = ""
        
        for line in content.split('\n'):
            if line.startswith('=== Conversation') or line.startswith('---'):
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = line + '\n'
            else:
                current_chunk += line + '\n'
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filtrage des chunks valides
        valid_chunks = [chunk for chunk in chunks if len(chunk) > 50]
        print(f"📊 {len(valid_chunks)} chunks valides extraits")
        
        # Génération des embeddings
        print("🧠 Génération des embeddings...")
        embeddings = model.encode(valid_chunks, show_progress_bar=True)
        
        # Sauvegarde PKL
        print("💾 Sauvegarde des données...")
        os.makedirs(INDEX_DIR, exist_ok=True)
        
        with open(INDEX_FILE, 'wb') as f:
            pickle.dump({
                'texts': valid_chunks,
                'embeddings': embeddings.tolist(),
                'model_name': 'all-MiniLM-L6-v2',
                'created_at': datetime.now().isoformat()
            }, f)
        
        # Création de l'index FAISS
        print("🏗️ Création de l'index FAISS...")
        embeddings_array = embeddings.astype('float32')
        index = faiss.IndexFlatL2(embeddings_array.shape[1])
        index.add(embeddings_array)
        
        faiss.write_index(index, FAISS_INDEX)
        
        print("✅ Régénération complète réussie!")
        print(f"📊 {len(valid_chunks)} documents indexés")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la régénération: {e}")
        logging.error(f"Erreur régénération: {e}")
        return False

def run_diagnostics():
    """Exécute un diagnostic complet et propose des solutions"""
    print("🔍 DIAGNOSTIC COMPLET SecondMind RAG")
    print("=" * 60)
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Répertoire de base: {BASE_DIR}")
    
    # 1. Vérification des fichiers
    files_status = check_files_status()
    
    # 2. Chargement des données
    texts, embeddings = load_and_verify_data()
    
    # 3. Vérification FAISS
    faiss_ok, faiss_index = verify_faiss_index(embeddings)
    
    # 4. Diagnostic et recommandations
    print("\n🎯 DIAGNOSTIC ET RECOMMANDATIONS")
    print("=" * 50)
    
    problems = []
    solutions = []
    
    if not files_status["Conversations source"]["exists"]:
        problems.append("❌ Fichier source de conversations manquant")
        solutions.append("1. Vérifiez le chemin du fichier conversations_extraites.txt")
    
    if not files_status["Index PKL"]["exists"]:
        problems.append("❌ Index PKL manquant")
        solutions.append("2. Exécutez la vectorisation pour créer l'index")
    
    if not faiss_ok:
        problems.append("❌ Index FAISS défaillant")
        solutions.append("3. Reconstruisez l'index FAISS")
    
    if texts is None or embeddings is None:
        problems.append("❌ Données corrompues ou illisibles")
        solutions.append("4. Régénération complète nécessaire")
    
    if not problems:
        print("✅ Tous les tests sont passés avec succès!")
        print("🎉 Votre système RAG est opérationnel")
        return True
    
    print("🚨 PROBLÈMES DÉTECTÉS:")
    for problem in problems:
        print(f"   {problem}")
    
    print("\n💡 SOLUTIONS RECOMMANDÉES:")
    for solution in solutions:
        print(f"   {solution}")
    
    return False

def interactive_menu():
    """Menu interactif de réparation"""
    while True:
        print("\n" + "="*60)
        print("🛠️  MENU DE RÉPARATION SECONDMIND RAG")
        print("="*60)
        print("1. 🔍 Diagnostic complet")
        print("2. 🔧 Réparer l'index FAISS")
        print("3. 🔄 Régénération complète")
        print("4. 📊 Vérifier les fichiers")
        print("5. 🧪 Test rapide")
        print("0. ❌ Quitter")
        print("="*60)
        
        try:
            choice = input("Votre choix: ").strip()
            
            if choice == "0":
                print("👋 Au revoir!")
                break
            elif choice == "1":
                run_diagnostics()
            elif choice == "2":
                texts, embeddings = load_and_verify_data()
                if texts and embeddings:
                    rebuild_faiss_index(texts, embeddings)
                else:
                    print("❌ Impossible de charger les données pour la réparation")
            elif choice == "3":
                confirm = input("⚠️ Régénération complète? Cela peut prendre du temps (o/N): ")
                if confirm.lower() in ['o', 'oui', 'y', 'yes']:
                    regenerate_from_source()
            elif choice == "4":
                check_files_status()
            elif choice == "5":
                texts, embeddings = load_and_verify_data()
                if texts and embeddings:
                    print(f"✅ Test réussi: {len(texts)} documents disponibles")
                else:
                    print("❌ Test échoué: données non disponibles")
            else:
                print("❌ Choix invalide")
                
        except KeyboardInterrupt:
            print("\n👋 Arrêt demandé par l'utilisateur")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")
            logging.error(f"Erreur menu: {e}")

if __name__ == "__main__":
    try:
        print("🚀 SecondMind RAG - Outil de diagnostic et réparation")
        print(f"📁 Répertoire de travail: {BASE_DIR}")
        
        # Création des dossiers si nécessaire
        os.makedirs(BASE_DIR, exist_ok=True)
        os.makedirs(INDEX_DIR, exist_ok=True)
        
        # Lancement du menu interactif
        interactive_menu()
        
    except Exception as e:
        print(f"❌ Erreur critique: {e}")
        logging.error(f"Erreur critique: {e}")
        input("Appuyez sur Entrée pour fermer...")
