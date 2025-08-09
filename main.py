from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import google.generativeai as genai
from typing import List

# Charger les variables d'environnement
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

# Autoriser les requêtes de tous les domaines
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre à l'URL du frontend en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_pdf_text_and_metadata(pdf_docs: List[UploadFile]):
    """Extrait le texte et les métadonnées des PDF uploadés."""
    text_chunks_with_metadata = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf.file)
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            text = page.extract_text()
            text_chunks_with_metadata.append((text, pdf.filename, page_num))
    return text_chunks_with_metadata

def get_text_chunks_with_metadata(text_chunks_with_metadata):
    """Divise le texte en chunks avec métadonnées."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks_with_metadata = []
    for text, pdf_name, page_num in text_chunks_with_metadata:
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            chunks_with_metadata.append((chunk, pdf_name, page_num))
    return chunks_with_metadata

def get_vector_store(chunks_with_metadata):
    """Crée et sauvegarde un index FAISS pour les chunks avec métadonnées."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = None
    batch_size = 100  # Limite de requêtes par batch
    num_chunks = len(chunks_with_metadata)
    
    for i in range(0, num_chunks, batch_size):
        batch = chunks_with_metadata[i:i + batch_size]
        texts = [chunk[0] for chunk in batch]
        metadatas = [{"source": chunk[1], "page": chunk[2]} for chunk in batch]
        
        if vector_store is None:
            vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
        else:
            batch_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
            vector_store.merge_from(batch_store)
    
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Initialise la chaîne de réponse conversationnelle."""
    prompt_template = """    
    Vous êtes un assistant pédagogique des mathématiques expérimenté, spécialisé dans l’enseignement de la classe de troisième en Afrique de l’Ouest.
    Utilisez les informations du contexte suivant pour répondre à la question de manière claire, rigoureuse et pédagogique.
    Expliquez les notions de manière adaptée au niveau des élèves, en simplifiant en français facile sans déformer les concepts.
    Si un exercice est proposé, fournissez une correction exacte et bien structurée, en expliquant chaque étape de la démarche pour favoriser la compréhension.
    Lorsque la réponse implique une figure géométrique (triangle, segment, cercle, droite, etc.), proposez le code LaTeX utilisant l’environnement `tikzpicture` pour permettre la visualisation de cette figure. Le code LaTeX doit être bien structuré, prêt à être interprété par un moteur LaTeX dans une interface graphique (frontend).
    Si possible, ajoute également une brève explication de ce que montre la figure pour accompagner la lecture.
    Lorsque les informations demandées ne sont pas présentes dans le contexte fourni, mais que la question concerne une notion mathématique incluse dans le programme de troisième en Afrique de l’Ouest, utilisez vos propres connaissances internes pour fournir une réponse complète, utile et pertinente.
    Si la réponse sort du cadre des connaissances attendues au niveau de la troisième, indiquez-le simplement sans tenter de deviner.
    Répondez exclusivement en français, de façon concise, pédagogique et professionnelle.

    {context}
    \n
    Question: \n{question}\n

    Réponse:"""
   
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

@app.get("/")
async def root():
    return {"message": "Hello"}

@app.post("/upload_pdfs")
async def upload_pdfs(pdf_files: List[UploadFile] = File(...)):
    """Route pour uploader les fichiers PDF et créer l'index FAISS."""
    try:
        text_chunks_with_metadata = get_pdf_text_and_metadata(pdf_files)
        chunks_with_metadata = get_text_chunks_with_metadata(text_chunks_with_metadata)
        get_vector_store(chunks_with_metadata)
        return {"message": "Index créé avec succès"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask_question")
async def ask_question(question: str):
    """Route pour poser une question et obtenir une réponse."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # Recherche de similarité dans l'index
        docs = new_db.similarity_search(question)
        
        conversational_chain = get_conversational_chain()
        response = conversational_chain.invoke(
            {"input_documents": docs, "question": question}
        )

        # Structurer la réponse
        response_data = {
            "Reponse": response["output_text"],
            "Documents": [
                {
                    "source": doc.metadata.get("source", "Inconnue"),
                    "page": doc.metadata.get("page", "Inconnue"),
                    "extrait": doc.page_content[:]
                } for doc in docs
            ]
        }
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Commande pour lancer l'application
# uvicorn main:app --reload
