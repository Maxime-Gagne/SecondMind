# -*- coding: utf-8 -*-
"""
Interface Gradio ONLINE pour SecondMind RAG
Version complète avec ouverture automatique du navigateur
"""

import os
import sys
import json
import gradio as gr
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

class SecondMindRAG:
    def __init__(self):
        self.vectorstore = None
        self.retriever = None
        self.conversations_lines = []
        self.metadata_info = {}
        self.setup_paths()
        
    def setup_paths(self):
        """Configuration des chemins absolus"""
        self.BASE_DIR = r"C:\Users\rag_personnel"
        self.DB_FAISS_PATH = os.path.join(self.BASE_DIR, "Logs", "vector_index_chatgpt")
        self.CONVERSATIONS_PATH = os.path.join(self.BASE_DIR, "Logs", "conversations_extraites.txt")
        
    def load_system(self):
        """Charge le système de recherche"""
        try:
            print("🔄 Chargement du système SecondMind...")

            if not os.path.exists(self.DB_FAISS_PATH):
                return False, f"❌ Index FAISS introuvable : {self.DB_FAISS_PATH}"
            
            if not os.path.exists(self.CONVERSATIONS_PATH):
                return False, f"❌ Fichier conversations introuvable : {self.CONVERSATIONS_PATH}"
            
            if not os.getenv("OPENAI_API_KEY"):
                return False, "❌ Variable d'environnement OPENAI_API_KEY non définie"
            
            metadata_file = os.path.join(self.DB_FAISS_PATH, "metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, "r", encoding="utf-8") as f:
                    self.metadata_info = json.load(f)
            
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                show_progress_bar=False
            )
            
            self.vectorstore = FAISS.load_local(
                self.DB_FAISS_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            
            return True, "✅ Système chargé avec succès"
        
        except Exception as e:
            return False, f"❌ Erreur lors du chargement : {str(e)}"

def lancer_interface():
    rag = SecondMindRAG()
    success, message = rag.load_system()
    if not success:
        print(message)
        return

    rag.retriever = rag.vectorstore.as_retriever()

    def répondre(question):
        if not rag.retriever:
            return "Système non prêt."
        try:
            docs = rag.retriever.get_relevant_documents(question)
            réponses = [doc.page_content for doc in docs]
            return "\n\n".join(réponses[:3]) if réponses else "Aucune réponse trouvée."
        except Exception as e:
            return f"Erreur lors de la récupération : {str(e)}"

    interface = gr.Interface(
        fn=répondre,
        inputs=gr.Textbox(lines=2, placeholder="Pose ta question ici..."),
        outputs="text",
        title="SecondMind RAG",
        description="Pose une question, reçois des réponses depuis ta base vectorielle locale."
    )

    interface.launch(share=True, inbrowser=True)

if __name__ == "__main__":
    lancer_interface()