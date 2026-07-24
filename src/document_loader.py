import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Cargar las variables de entorno (.env)
load_dotenv()

def build_vector_store(pdf_path: str, persist_dir: str = "chroma_db"):
    """
    Lee el PDF de la ley, lo divide en fragmentos
    y genera la base de datos vectorial local con HuggingFace Embeddings.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ No se encontró el archivo en la ruta: {pdf_path}")

    print("📄 Cargando el documento de la Ley...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ Se cargaron {len(documents)} páginas del PDF.")

    print("✂️ Fragmentando el texto (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "Artículo ", "ARTÍCULO ", " "]
    )
    docs = text_splitter.split_documents(documents)
    print(f"✅ Se crearon {len(docs)} fragmentos de texto.")

    print("🧠 Generando Embeddings locales y guardando en ChromaDB...")
    
    # Modelo local ultraligero y muy rápido
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"🎉 Base de datos vectorial guardada con éxito en la carpeta '{persist_dir}'.")
    return vectorstore

if __name__ == "__main__":
    path_pdf = os.path.join("data", "ley_economia_circular_mexico.pdf")
    build_vector_store(path_pdf)