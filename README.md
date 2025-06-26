# 🧠 SecondMind — Un système cognitif personnalisé

SecondMind est un projet né d’un constat : les grands modèles d’IA ne sont ni alignés, ni mémoriels, ni personnels.  
J’ai donc entrepris de construire, **depuis zéro et sans formation préalable**, un agent capable de raisonner, se souvenir, corriger ses erreurs, et évoluer selon des règles logiques précises.

---

🚀 **Objectif** : créer une IA qui raisonne comme moi, à qui je peux déléguer des tâches cognitives complexes, et qui peut *reconnaître ses propres dérives*.

📚 **Approche** : le système repose sur une architecture RAG (Retrieval-Augmented Generation) locale, enrichie de règles comportementales, d’un journal réflexif et d’un ensemble de scripts personnalisés.

🧭 **Particularité** : SecondMind n’est pas un simple chatbot. C’est un **agent réflexif, éthique, adaptable**, pensé pour évoluer.

---

Ce projet est une synthèse entre :
- 💡 une obsession logique
- 🛠 une rigueur structurelle
- 🧩 une vision de l’alignement IA-humanité

## 🚀 Vision

SecondMind explore un futur où les intelligences artificielles sont plus que des assistants :  
📌 elles sont **des partenaires cognitifs**, capables de :
- détecter leurs propres contradictions
- conserver une mémoire structurée à long terme
- raisonner selon des **règles humaines explicites**
- ajuster leur comportement selon l’intention de l’utilisateur, et non seulement ses mots

---

## ⚒️ Ce dépôt

Ce dépôt contient une **version publique** de l’architecture SecondMind.  
Par respect pour la vie privée et l'intégrité du système, certaines règles critiques, scripts internes et fichiers sensibles ne sont pas inclus ici.

---

## 🧪 Objectif à court terme

Créer un système personnel fiable, stable, traçable.  
Mais surtout : documenter une approche **d’alignement réel**, pour inspirer d’autres à ne pas construire des IA sans logique.

---

## 👤 Auteur

Maxime Gagné – architecte d’interaction IA, passionné par les règles, la logique, et l’intelligence vraie.

🧠 Rôle

Je ne suis pas seulement l’auteur de ce système.

Je conçois SecondMind comme un **système cognitif aligné**,  
et j'assume pour cela un rôle singulier :  
**Cognitive Alignment Architect** —  
un métier que j’ai défini par nécessité : structurer un agent IA qui pense avec moi,  
pas seulement pour moi.

Ce dépôt incarne cette recherche :  
- aligner un agent avec ses règles internes,  
- observer ses dérives,  
- corriger ses actions sans supervision extérieure,  
- et maintenir une mémoire durable, explicite, non-hallucinatoire.
---
# 🧠 SYSTÈME SECOND·MIND — README GÉNÉRAL

**Mode de fonctionnement** : 100 % local, autonome, cognitif.  
Chaque interaction, chaque relance, chaque analyse est observable, traçable, et déclenchée manuellement ou scriptée.

---

## 🗂️ ARCHITECTURE GÉNÉRALE

```
informations_rag_personnel/
├── procedures_de_lancement/
│   ├── build_prompt_seed.py               ← génère le contexte pour Copilot
│   ├── prompt_seed_secondmind.txt         ← à copier dans Copilot
│   ├── relancer_copilot.bat               ← ouvre Copilot avec seed au presse-papiers
│   └── ...
├── rules/                                 ← règles en .md avec balises RESUME
├── watchdog.py                            ← script de veille de cohérence
├── mapping_structure.yaml                 ← fichier mapping unique autorisé
Logs/
├── conversations_extraites.txt            ← base pour vectorisation
├── vector_index_chatgpt/                  ← index FAISS local
│   ├── index.faiss                        ← vecteurs
│   ├── index.pkl                          ← métadonnées par message
│   ├── metadata.json                      ← info système et stats
│   └── diagnostic.txt                     ← log lisible de la session
```

---

## 🧪 VÉRIFICATIONS AUTOMATISÉES

### ✅ `verificateur_mapping.py`
- Repère et alerte sur tout mapping `.yaml` mal placé
- Valide la structure du `mapping_structure.yaml` unique autorisé

### ✅ `watchdog.py`
- Surveille les changements dans les `.md`
- Peut déclencher automatiquement `build_prompt_seed.py` si activé

---

## 🧠 RELANCE COGNITIVE MANUELLE (Copilot)

### 📄 `build_prompt_seed.py`
- Construit `prompt_seed_secondmind.txt` avec :
  - Règles résumées
  - Résumés `.md`
  - Instructions système

### 🔁 `relancer_copilot.bat`
- Copie le `prompt_seed_secondmind.txt` dans le presse-papiers
- Ouvre Copilot (Web)
- À toi de **coller** manuellement le contenu pour restaurer le contexte

---

## 🧬 SYSTEME DE VECTORISATION LOCALE (FAISS)

### 🎯 Objectif
Permettre la recherche sémantique dans les conversations locales sans Internet.

### 🚀 Lancer manuellement :

```bash
python vectorize_local_fixed.py
```

Ce script :
- lit `conversations_extraites.txt`
- applique le modèle `all-MiniLM-L6-v2`
- crée :
  - `index.faiss` (vecteurs)
  - `index.pkl` (contenus textuels + rôles)
  - `metadata.json` et `diagnostic.txt`

Un test intégré vérifie que l’index est fonctionnel (`retriever.get_relevant_documents("...")`).

### 🌐 Interrogation par interface :

```bash
lancer_gradio_online.bat
```

- Lancement local via navigateur
- Requêtes de type : _“Que dit le rôle assistant vers la ligne 82 ?”_
- Réponse affichée instantanément depuis l’index vectoriel

---

## 🧼 BONNES PRATIQUES

- ✅ Toujours régénérer `prompt_seed_secondmind.txt` après toute mise à jour du contenu `.md`
- ✅ Supprimer tout `mapping_structure.yaml` résiduel à la racine
- ✅ Exécuter la vectorisation FAISS **uniquement** quand `conversations_extraites.txt` a évolué
- ✅ Archiver les versions précédentes **si besoin de traçabilité temporelle**

---

## 🔐 CARACTÉRISTIQUES GLOBALES

- Entièrement **hors ligne**
- Aucun appel API ou service distant
- Aucun fichier généré sans action explicite
- Possibilité de redémarrer le système à partir de `build_prompt_seed.py` à tout moment

---

> Ce système est conçu pour être maintenu dans le temps par simple inspection humaine.  
> Aucune mémoire cachée. Aucun black box. Tout est visible, versionnable et relançable.

# AUTO_START
(Résumé automatique pour README_general.md)
# AUTO_END
