import os
import time
import pandas as pd
from dotenv import load_dotenv

# Evitar alertas de paralelismo, barras de progreso y warnings de Hugging Face Hub
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Cargar variables de entorno del archivo .env
load_dotenv(override=True)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

def cargar_vectorstore(persist_dir: str = "chroma_db"):
    """Carga la base de datos vectorial existente."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

def auditar_mipyme(csv_path: str):
    """
    Lee los datos del CSV de la Mipyme, consulta la Ley
    y genera un diagnóstico detallado con Gemini.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ No se encontró el CSV en: {csv_path}")

    # 1. Obtener y validar la API Key de las variables de entorno
    api_key_env = os.getenv("GOOGLE_API_KEY")
    if not api_key_env:
        raise ValueError("❌ No se encontró la variable GOOGLE_API_KEY en el archivo .env")

    print("🔍 Cargando base de datos vectorial...")
    vectorstore = cargar_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 2. Inicializar Gemini (nota: el argumento es google_api_key en minúsculas)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=api_key_env,
        temperature=0.2,
        max_retries=3
    )

    # 3. Leer datos de la Mipyme desde el CSV
    df = pd.read_csv(csv_path)
    datos_empresa = df.to_string(index=False)

    # 4. Buscar contexto legal relevante en ChromaDB
    print("📚 Consultando el marco legal relevante...")
    query = "requisitos, obligaciones, plan de manejo y porcentajes de reciclaje para empresas"
    docs_relevantes = retriever.invoke(query)
    contexto_legal = "\n\n".join([doc.page_content for doc in docs_relevantes])

    # 5. Configurar el Prompt
    prompt_template = """
    Eres un Agente Auditor Oficial especializado en la nueva Ley General de Economía Circular de México.
    Tu objetivo es realizar un diagnóstico legal riguroso pero constructivo para una Mipyme.

    CONTEXTO LEGAL DE LA LEY:
    {contexto_legal}

    DATOS EVALUADOS DE LA MIPYME:
    {datos_empresa}

    Por favor, genera un REPORTE DE AUDITORÍA en formato Markdown estructurado así:

    # 📋 Reporte de Evaluación de Cumplimiento Ambiental
    ## 1. Diagnóstico General
    Un resumen ejecutivo del nivel de preparación actual de la Mipyme.

    ## 2. Análisis Punto por Punto
    Evalúa cada fila del reporte de la Mipyme comparándolo con el marco legal disponible:
    - Estado de cumplimiento (Cumple / Cumple Parcialmente / No Cumple).
    - Fundamento legal u observación técnica.

    ## 3. Plan de Acción y Medidas Correctivas
    Acciones concretas priorizadas para subsanar los incumplimientos detectados.

    ## 4. Dictamen Final y Certificación
    Indica explícitamente si la Mipyme es **APTA** o **NO APTA** para recibir el Distintivo/Certificado de Economía Circular en este momento y por qué.
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["contexto_legal", "datos_empresa"]
    )

    print("🧠 Generando diagnóstico con Gemini...\n")
    print("="*50 + "\n")

    time.sleep(1)

    chain = prompt | llm
    
    return chain.stream({
        "contexto_legal": contexto_legal,
        "datos_empresa": datos_empresa
    })

if __name__ == "__main__":
    csv_file = os.path.join("data", "template_evaluacion.csv")
    
    for chunk in auditar_mipyme(csv_file):
        print(chunk.content, end="", flush=True)
        
    print("\n\n" + "="*50)