# -*- coding: utf-8 -*-
"""
Vectorisation LOCALE avec HuggingFace Embeddings
Version corrigée avec chemins absolus et améliorations
"""
import os
import sys
import json
import pickle
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

def nettoyer_ligne(texte):
    """Nettoie et standardise une ligne de texte"""
    return texte.strip().replace("\n", " ").replace("\r", "").replace("  ", " ").strip()

def extraire_role_et_contenu(ligne):
    """Extrait le rôle et le contenu d'une ligne de conversation"""
    ligne_nettoyee = nettoyer_ligne(ligne)
    if not ligne_nettoyee:
        return None, None
    
    # Détection du format "role|contenu" ou similaire
    if "|" in ligne_nettoyee:
        parts = ligne_nettoyee.split("|", 1)
        role = parts[0].strip().lower()
        contenu = parts[1].strip()
    elif ligne_nettoyee.lower().startswith(("user:", "assistant:", "human:", "ai:")):
        parts = ligne_nettoyee.split(":", 1)
        role = parts[0].strip().lower()
        contenu = parts[1].strip() if len(parts) > 1 else ""
    else:
        role = "unknown"
        contenu = ligne_nettoyee
    
    return role, contenu

def main():
    print("🚀 DÉMARRAGE DE LA VECTORISATION LOCALE (HuggingFace)")
    print("=" * 65)
    
    # === CHEMINS ABSOLUS FIXES ===
    BASE_DIR = r"C:\Users\rag_personnel"
    DATA_PATH = os.path.join(BASE_DIR, "Logs", "conversations_extraites.txt")
    DB_FAISS_PATH = os.path.join(BASE_DIR, "Logs", "vector_index_chatgpt")
    
    print(f"📁 Fichier source : {DATA_PATH}")
    print(f"📁 Dossier index : {DB_FAISS_PATH}")
    
    # === VÉRIFICATIONS PRÉLIMINAIRES ===
    if not os.path.exists(DATA_PATH):
        print(f"❌ ERREUR : Fichier source introuvable : {DATA_PATH}")
        input("Appuyez sur Entrée pour fermer...")
        sys.exit(1)
    
    # Créer le dossier de destination si nécessaire
    os.makedirs(DB_FAISS_PATH, exist_ok=True)
    
    # === LECTURE ET TRAITEMENT DES DONNÉES ===
    print("\n📖 Lecture du fichier source...")
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
    except Exception as e:
        print(f"❌ ERREUR lors de la lecture : {e}")
        input("Appuyez sur Entrée pour fermer...")
        sys.exit(1)
    
    print(f"✅ {len(all_lines)} lignes lues")
    
    # === CRÉATION DES DOCUMENTS ===
    print("\n🔄 Traitement des documents...")
    docs = []
    stats = {"user": 0, "assistant": 0, "unknown": 0, "empty": 0}
    
    for i, line in enumerate(all_lines):
        role, contenu = extraire_role_et_contenu(line)
        
        if not contenu:
            stats["empty"] += 1
            continue
        
        # Mise à jour des statistiques
        if role in stats:
            stats[role] += 1
        else:
            stats["unknown"] += 1
        
        # Création du document
        metadata = {
            "source": "conversations_extraites.txt",
            "ligne": i + 1,
            "role": role,
            "longueur": len(contenu),
            "timestamp": datetime.now().isoformat()
        }
        
        doc = Document(page_content=contenu, metadata=metadata)
        docs.append(doc)
    
    print(f"✅ {len(docs)} documents créés")
    print(f"📊 Statistiques : {stats}")
    
    if not docs:
        print("❌ ERREUR : Aucun document valide trouvé")
        input("Appuyez sur Entrée pour fermer...")
        sys.exit(1)
    
    # === INITIALISATION DES EMBEDDINGS LOCAUX ===
    print("\n🧠 Initialisation des embeddings locaux...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    try:
        # Méthode 1 : Utiliser HuggingFaceEmbeddings (recommandé pour LangChain)
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # Utiliser 'cuda' si GPU disponible
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✅ Embeddings HuggingFace initialisés")
        
        # Test rapide des embeddings
        test_embedding = embeddings.embed_query("test")
        print(f"✅ Test d'embedding réussi (dimension: {len(test_embedding)})")
        
    except Exception as e:
        print(f"❌ ERREUR lors de l'initialisation des embeddings : {e}")
        print("💡 Essayez d'installer : pip install sentence-transformers")
        input("Appuyez sur Entrée pour fermer...")
        sys.exit(1)
    
    # === VECTORISATION ===
    print("\n🔄 Vectorisation en cours...")
    try:
        # Traitement par batch pour gérer la mémoire
        batch_size = 50  # Plus petit pour le local
        
        if len(docs) > batch_size:
            print(f"📦 Traitement par batch de {batch_size} documents")
            
            # Premier batch pour initialiser l'index
            first_batch = docs[:batch_size]
            index = FAISS.from_documents(first_batch, embeddings)
            print(f"✅ Premier batch traité : {len(first_batch)} documents")
            
            # Batches suivants
            for i in range(batch_size, len(docs), batch_size):
                batch = docs[i:i+batch_size]
                batch_index = FAISS.from_documents(batch, embeddings)
                index.merge_from(batch_index)
                print(f"✅ Batch {i//batch_size + 1} traité : {len(batch)} documents")
                
                # Affichage du progrès
                progress = min(i + batch_size, len(docs))
                percentage = (progress / len(docs)) * 100
                print(f"📊 Progrès : {progress}/{len(docs)} ({percentage:.1f}%)")
        else:
            index = FAISS.from_documents(docs, embeddings)
        
        print("✅ Vectorisation terminée")
        
    except Exception as e:
        print(f"❌ ERREUR lors de la vectorisation : {e}")
        input("Appuyez sur Entrée pour fermer...")
        sys.exit(1)
    
    # === SAUVEGARDE ===
    print("\n💾 Sauvegarde de l'index...")
    try:
        index.save_local(DB_FAISS_PATH)
        print("✅ Index FAISS sauvegardé")
        
        # Sauvegarde des métadonnées supplémentaires
        metadata_file = os.path.join(DB_FAISS_PATH, "metadata.json")
        metadata_info = {
            "created_at": datetime.now().isoformat(),
            "source_file": DATA_PATH,
            "total_documents": len(docs),
            "embedding_model": model_name,
            "embedding_type": "local_huggingface",
            "stats": stats,
            "version": "2.0"
        }
        
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata_info, f, indent=2, ensure_ascii=False)
        
        print("✅ Métadonnées sauvegardées")
        
        # Sauvegarde des métadonnées pour compatibilité avec les anciens scripts
        metadatas_list = []
        for doc in docs:
            metadatas_list.append({
                "ligne_originale": doc.metadata["ligne"],
                "role": doc.metadata["role"],
                "texte_complet": doc.page_content,
                "longueur": doc.metadata["longueur"]
            })
        
        with open(os.path.join(DB_FAISS_PATH, "index.pkl"), "wb") as f:
            pickle.dump(metadatas_list, f)
        
        print("✅ Métadonnées de compatibilité sauvegardées")
        
    except Exception as e:
        print(f"❌ ERREUR lors de la sauvegarde : {e}")
        input("Appuyez sur Entrée pour fermer...")
        sys.exit(1)
    
    # === CRÉATION DU DIAGNOSTIC ===
    diagnostic_file = os.path.join(DB_FAISS_PATH, "diagnostic.txt")
    try:
        with open(diagnostic_file, "w", encoding="utf-8") as f:
            f.write(f"=== DIAGNOSTIC VECTORISATION LOCALE ===\n")
            f.write(f"Date de création : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Fichier source : {DATA_PATH}\n")
            f.write(f"Modèle d'embedding : {model_name}\n")
            f.write(f"Nombre total de documents : {len(docs)}\n")
            f.write(f"Statistiques par rôle :\n")
            for role, count in stats.items():
                f.write(f"  - {role} : {count}\n")
            f.write(f"\nIndex sauvegardé dans : {DB_FAISS_PATH}\n")
            f.write(f"✅ Vectorisation réussie\n")
        
        print("✅ Diagnostic créé")
        
    except Exception as e:
        print(f"⚠️  Avertissement : Impossible de créer le diagnostic : {e}")
    
    # === TEST DE L'INDEX ===
    print("\n🧪 Test de l'index créé...")
    try:
        # Test de rechargement
        test_index = FAISS.load_local(
            DB_FAISS_PATH, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Test de recherche
        retriever = test_index.as_retriever(search_kwargs={"k": 3})
        test_results = retriever.get_relevant_documents("conversation")
        
        print("✅ Index testé avec succès")
        print(f"🔍 Test de recherche : {len(test_results)} résultats trouvés")
        
        if test_results:
            doc = test_results[0]
            print(f"📄 Exemple de résultat :")
            print(f"   Rôle : {doc.metadata.get('role', 'N/A')}")
            print(f"   Ligne : {doc.metadata.get('ligne', 'N/A')}")
            print(f"   Contenu : {doc.page_content[:100]}...")
        
    except Exception as e:
        print(f"⚠️  Avertissement lors du test : {e}")
    
    # === RÉSUMÉ FINAL ===
    print("\n" + "=" * 65)
    print("🎉 VECTORISATION LOCALE TERMINÉE AVEC SUCCÈS !")
    print(f"✅ {len(docs)} documents vectorisés")
    print(f"📁 Index sauvegardé dans : {DB_FAISS_PATH}")
    print(f"🧠 Modèle utilisé : {model_name}")
    print("💡 Avantages du modèle local :")
    print("   - Aucun coût d'API")
    print("   - Fonctionne hors ligne")
    print("   - Données privées")
    print("🚀 Vous pouvez maintenant lancer l'interface Gradio LOCAL")
    print("=" * 65)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Vectorisation interrompue par l'utilisateur")
    except Exception as e:
        print(f"\n💥 ERREUR CRITIQUE : {e}")
    finally:
        print("\n⏸️  Appuyez sur Entrée pour fermer...")
        input()