from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings  # Ou HuggingFaceEmbeddings selon le moteur
import os
import json

# === Chemin de l'index
DB_FAISS_PATH = r"C:\Users\rag_personnel\Logs\vector_index_chatgpt"

# === Chargement des embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # ou supprime ce paramètre si HuggingFace
)

# === Chargement de l'index FAISS
print("🔍 Chargement de l'index...")
try:
    index = FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    print(f"❌ Erreur de chargement : {e}")
    exit()

# === Recherche test
query = "test"
print(f"\n🧪 Requête test : {query}")
docs = index.similarity_search(query, k=3)

if not docs:
    print("⚠️ Aucun document retourné.")
else:
    print(f"✅ {len(docs)} document(s) trouvés :")
    for doc in docs:
        print("—" * 50)
        print(f"Rôle : {doc.metadata.get('role')}")
        print(f"Ligne : {doc.metadata.get('ligne')}")
        print(f"Contenu : {doc.page_content[:120]}...")

# === Lecture des métadonnées globales
meta_path = os.path.join(DB_FAISS_PATH, "metadata.json")
if os.path.exists(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        print("\n📊 Métadonnées de session :")
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
else:
    print("ℹ️ Aucun fichier metadata.json trouvé.")
