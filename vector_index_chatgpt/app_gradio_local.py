# -*- coding: utf-8 -*-
"""
Interface Gradio LOCALE pour SecondMind RAG
Version corrigée avec chemins absolus et améliorations
"""

import os
import sys
import json
import gradio as gr
from datetime import datetime
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import logging

# Configuration des chemins absolus
BASE_DIR = r"C:\Users\rag_personnel\Logs"
INDEX_DIR = os.path.join(BASE_DIR, "vector_index_chatgpt")
INDEX_FILE = os.path.join(INDEX_DIR, "index.pkl")
FAISS_INDEX = os.path.join(INDEX_DIR, "index.faiss")
CONVERSATIONS_FILE = os.path.join(BASE_DIR, "conversations_extraites.txt")
LOG_FILE = os.path.join(BASE_DIR, "gradio_local.log")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class LocalRAGSystem:
    def __init__(self):
        self.model = None
        self.vectorstore = None
        self.texts = None
        self.embeddings = None
        
    def initialize(self):
        """Initialise le système RAG local"""
        try:
            logging.info("🚀 Démarrage du système RAG LOCAL...")
            
            # Chargement du modèle
            logging.info("📥 Chargement du modèle SentenceTransformer...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Vérification des fichiers
            if not os.path.exists(INDEX_FILE):
                raise FileNotFoundError(f"Index introuvable : {INDEX_FILE}")
            if not os.path.exists(FAISS_INDEX):
                raise FileNotFoundError(f"Index FAISS introuvable : {FAISS_INDEX}")
                
            # Chargement de l'index
            logging.info("📂 Chargement des données vectorisées...")
            with open(INDEX_FILE, 'rb') as f:
                data = pickle.load(f)
                self.texts = data['texts']
                self.embeddings = data['embeddings']
            
            # Chargement de l'index FAISS
            self.vectorstore = faiss.read_index(FAISS_INDEX)
            
            logging.info(f"✅ Système initialisé avec {len(self.texts)} documents")
            return True, f"✅ Système prêt avec {len(self.texts)} documents"
            
        except Exception as e:
            error_msg = f"❌ Erreur d'initialisation : {str(e)}"
            logging.error(error_msg)
            return False, error_msg
    
    def search_similar(self, query, k=5):
        """Recherche de documents similaires"""
        try:
            if not self.model or not self.vectorstore:
                return [], "❌ Système non initialisé"
            
            # Vectorisation de la requête
            query_embedding = self.model.encode([query])
            
            # Recherche dans l'index FAISS
            distances, indices = self.vectorstore.search(
                query_embedding.astype('float32'), k
            )
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.texts):
                    results.append({
                        'rank': i + 1,
                        'text': self.texts[idx],
                        'score': float(1 - distance),  # Conversion en similarité
                        'distance': float(distance)
                    })
            
            return results, f"✅ {len(results)} résultats trouvés"
            
        except Exception as e:
            error_msg = f"❌ Erreur de recherche : {str(e)}"
            logging.error(error_msg)
            return [], error_msg

# Instance globale
rag_system = LocalRAGSystem()

def initialize_system():
    """Interface pour initialiser le système"""
    success, message = rag_system.initialize()
    return message

def search_interface(query, num_results=5):
    """Interface de recherche pour Gradio"""
    if not query.strip():
        return "⚠️ Veuillez saisir une requête"
    
    try:
        results, status = rag_system.search_similar(query, k=num_results)
        
        if not results:
            return f"{status}\n\nAucun résultat trouvé pour : '{query}'"
        
        # Formatage des résultats
        output = f"🔍 **Recherche :** {query}\n"
        output += f"📊 **Résultats :** {len(results)}\n"
        output += "─" * 50 + "\n\n"
        
        for result in results:
            output += f"**#{result['rank']} | Score: {result['score']:.3f}**\n"
            output += f"{result['text'][:500]}{'...' if len(result['text']) > 500 else ''}\n"
            output += "─" * 30 + "\n\n"
        
        # Log de la recherche
        logging.info(f"Recherche effectuée : '{query}' -> {len(results)} résultats")
        
        return output
        
    except Exception as e:
        error_msg = f"❌ Erreur lors de la recherche : {str(e)}"
        logging.error(error_msg)
        return error_msg

def get_system_stats():
    """Affiche les statistiques du système"""
    try:
        stats = f"📊 **STATISTIQUES SYSTÈME**\n\n"
        
        # Vérification des fichiers
        files_status = []
        files_to_check = [
            ("Conversations", CONVERSATIONS_FILE),
            ("Index PKL", INDEX_FILE),
            ("Index FAISS", FAISS_INDEX)
        ]
        
        for name, path in files_to_check:
            if os.path.exists(path):
                size = os.path.getsize(path) / (1024*1024)  # MB
                files_status.append(f"✅ {name}: {size:.1f} MB")
            else:
                files_status.append(f"❌ {name}: Non trouvé")
        
        stats += "\n".join(files_status)
        
        # Informations système
        if rag_system.texts:
            stats += f"\n\n📚 Documents chargés : {len(rag_system.texts)}"
            stats += f"\n🧠 Modèle : all-MiniLM-L6-v2"
            stats += f"\n📍 Mode : LOCAL (HuggingFace)"
        
        stats += f"\n\n⏰ Dernière vérification : {datetime.now().strftime('%H:%M:%S')}"
        
        return stats
        
    except Exception as e:
        return f"❌ Erreur lors de la récupération des stats : {str(e)}"

# Interface Gradio
def create_interface():
    """Crée l'interface Gradio"""
    
    with gr.Blocks(
        title="SecondMind RAG - Version LOCALE",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .main-header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        """
    ) as interface:
        
        # En-tête
        gr.HTML("""
        <div class="main-header">
            <h1>🧠 SecondMind RAG System</h1>
            <p>Version LOCALE avec HuggingFace Embeddings</p>
        </div>
        """)
        
        with gr.Tab("🔍 Recherche"):
            with gr.Row():
                with gr.Column(scale=3):
                    query_input = gr.Textbox(
                        label="Votre question",
                        placeholder="Tapez votre question ici...",
                        lines=2
                    )
                    num_results = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Nombre de résultats"
                    )
                    search_btn = gr.Button("🔍 Rechercher", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    init_btn = gr.Button("🚀 Initialiser le système", variant="secondary")
                    stats_btn = gr.Button("📊 Voir les stats", variant="secondary")
            
            results_output = gr.Textbox(
                label="Résultats",
                lines=20,
                max_lines=30,
                show_copy_button=True
            )
        
        with gr.Tab("📊 Statistiques"):
            stats_output = gr.Textbox(
                label="Informations système",
                lines=15,
                show_copy_button=True
            )
            refresh_stats_btn = gr.Button("🔄 Actualiser", variant="secondary")
        
        with gr.Tab("ℹ️ Aide"):
            gr.Markdown("""
            ## 🛠️ Guide d'utilisation
            
            ### Étapes pour utiliser le système :
            
            1. **Initialisation** : Cliquez sur "🚀 Initialiser le système"
            2. **Recherche** : Tapez votre question et cliquez sur "🔍 Rechercher"
            3. **Ajustements** : Modifiez le nombre de résultats selon vos besoins
            
            ### 📁 Fichiers requis :
            - `conversations_extraites.txt` : Conversations source
            - `vector_index_chatgpt/index.pkl` : Index vectoriel
            - `vector_index_chatgpt/index.faiss` : Index FAISS
            
            ### 🔧 Fonctionnalités :
            - **Recherche sémantique** : Trouve des réponses pertinentes
            - **Score de pertinence** : Évalue la qualité des résultats
            - **Interface intuitive** : Simple d'utilisation
            - **Statistiques** : Suivi des performances
            
            ### ⚠️ Notes importantes :
            - Ce système fonctionne en mode LOCAL (sans API)
            - Les données restent sur votre machine
            - Première utilisation : attendez le téléchargement du modèle
            """)
        
        # Événements
        search_btn.click(
            search_interface,
            inputs=[query_input, num_results],
            outputs=[results_output]
        )
        
        init_btn.click(
            initialize_system,
            outputs=[results_output]
        )
        
        stats_btn.click(
            get_system_stats,
            outputs=[results_output]
        )
        
        refresh_stats_btn.click(
            get_system_stats,
            outputs=[stats_output]
        )
        
        # Auto-load des stats au démarrage
        interface.load(get_system_stats, outputs=[stats_output])
    
    return interface

if __name__ == "__main__":
    try:
        print("🚀 Démarrage de SecondMind RAG - Version LOCALE")
        print(f"📁 Répertoire de travail : {BASE_DIR}")
        
        # Vérification des dossiers
        os.makedirs(BASE_DIR, exist_ok=True)
        os.makedirs(INDEX_DIR, exist_ok=True)
        
        # Création et lancement de l'interface
        interface = create_interface()
        
        print("🌐 Lancement de l'interface web...")
        interface.launch(
            server_name="127.0.0.1",
            server_port=7861,  # Port différent pour éviter les conflits
            share=False,
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        print(f"❌ Erreur critique : {e}")
        logging.error(f"Erreur critique : {e}")
        input("Appuyez sur Entrée pour fermer...")
