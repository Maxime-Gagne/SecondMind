# Version DEBUG avec chargement FAISS sécurisé
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Chargement FAISS avec gestion d'erreur
DB_FAISS_PATH = "Logs/vector_index_chatgpt"
embedding = OpenAIEmbeddings()

print("🔄 Tentative de chargement de l'index FAISS...")

try:
    # Méthode 1 : Chargement normal
    vectorstore = FAISS.load_local(
        DB_FAISS_PATH, 
        embedding, 
        allow_dangerous_deserialization=True
    )
    print("✅ Index FAISS chargé avec succès (méthode normale)")
    
except Exception as e1:
    print(f"❌ Méthode 1 échouée : {e1}")
    try:
        # Méthode 2 : Chargement alternatif
        import faiss
        import pickle
        
        # Chargement manuel des composants
        index_path = os.path.join(DB_FAISS_PATH, "index.faiss")
        pkl_path = os.path.join(DB_FAISS_PATH, "index.pkl")
        
        # Charger l'index FAISS
        index = faiss.read_index(index_path)
        print(f"✅ Index FAISS brut chargé : {index.ntotal} vecteurs")
        
        # Charger les métadonnées
        with open(pkl_path, 'rb') as f:
            store_data = pickle.load(f)
        print(f"✅ Métadonnées chargées : {len(store_data)} entrées")
        
        # Créer le vectorstore manuellement
        vectorstore = FAISS(
            embedding_function=embedding,
            index=index,
            docstore=store_data.get('docstore', {}),
            index_to_docstore_id=store_data.get('index_to_docstore_id', {})
        )
        print("✅ Vectorstore reconstruit manuellement")
        
    except Exception as e2:
        print(f"❌ Toutes les méthodes ont échoué : {e2}")
        print("🔍 Contenu du dossier :")
        for file in os.listdir(DB_FAISS_PATH):
            file_path = os.path.join(DB_FAISS_PATH, file)
            size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"  - {file}: {size:.1f} MB")
        exit(1)

# Test du retriever
print("\n🔍 TEST DU RETRIEVER :")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

test_query = "douleur main droite"
try:
    docs = retriever.get_relevant_documents(test_query)
    print(f"📄 DOCUMENTS TROUVÉS ({len(docs)}) :")
    for i, doc in enumerate(docs):
        print(f"\n--- Document {i+1} ---")
        print(f"Contenu: {doc.page_content[:200]}...")
        if hasattr(doc, 'metadata'):
            print(f"Metadata: {doc.metadata}")
except Exception as e:
    print(f"❌ Erreur lors de la recherche : {e}")
    exit(1)

# Configuration du prompt
debug_prompt_template = """Utilise les informations suivantes pour répondre à la question de l'utilisateur.
Si tu ne connais pas la réponse, dis-le clairement. Ne tente pas d'inventer une réponse.

Contexte : {context}
---
Question : {question}
Réponse : """

prompt = PromptTemplate(
    template=debug_prompt_template,
    input_variables=["context", "question"]
)

# Modèle LLM
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# Chaîne QA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=False
)

print("\n" + "="*50)
print("✅ RAG en mode DEBUG - Tapez 'exit' pour quitter")
print("="*50)

while True:
    query = input("\nVous: ")
    if query.lower() in ["exit", "quit", "q"]:
        print("👋 Fin du RAG.")
        break
    
    try:
        result = qa_chain.run(query)
        print(f"\n📝 RÉPONSE : {result}")
        
    except Exception as e:
        print(f"❌ Erreur lors de la génération : {e}")
