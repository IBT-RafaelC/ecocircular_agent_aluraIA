import os
import pandas as pd
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# 1. Cargar variables de entorno
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("⚠️ No se encontró la GROQ_API_KEY en el archivo .env")

# 2. Inicializar el LLM (Groq - Llama 3.3 70B)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    streaming=True
)

# 3. Configurar la base de datos vectorial (ChromaDB)
# Ajusta la ruta a 'chroma_db' según la ubicación de tu proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_db")

embedding_function = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"local_files_only": True}
)

vector_store = Chroma(
    persist_directory=CHROMA_DIR if os.path.exists(CHROMA_DIR) else "chroma_db",
    embedding_function=embedding_function
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


def generar_diagnostico_stream(datos_empresa: str):
    """
    Función generadora para consumir el flujo de respuesta (streaming)
    tanto en la terminal como en la interfaz gráfica con Streamlit.
    """
    # Recuperación del contexto legal relevante
    docs = retriever.invoke(datos_empresa)
    contexto = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""Eres un auditor especialista en la Ley General de Economía Circular de México. 
Tu objetivo es evaluar los datos operativos de la MiPyME proporcionada y redactar un informe de cumplimiento claro, profesional y estructurado en Markdown.

Utiliza ÚNICAMENTE la siguiente información del marco legal para respaldar tus recomendaciones y observaciones:

<marco_legal>
{contexto}
</marco_legal>

Datos operativos de la empresa a evaluar:
<datos_empresa>
{datos_empresa}
</datos_empresa>

Estructura tu reporte de la siguiente manera:
1. **Resumen Ejecutivo**: Breve diagnóstico global de la situación de la MiPyME.
2. **Evaluación de Cumplimiento**: Lista detallada de hallazgos comparando los datos operativos con el marco legal (menciona artículos o secciones relevantes si aplican).
3. **Plan de Acción Sugerido**: Recomendaciones concretas, viables y priorizadas para mejorar el cumplimiento circular.
"""

    # Retorna cada fragmento a medida que Groq lo genera
    for chunk in llm.stream(prompt):
        yield chunk.content


def generar_diagnostico_cli(datos_empresa: str):
    """
    Función auxiliar para ejecutar y visualizar el resultado directamente en la consola.
    """
    print("📚 Consultando el marco legal relevante...")
    print("🧠 Generando diagnóstico con Llama 3.3 (Groq)...\n")
    print("=" * 50 + "\n")

    full_text = ""
    for chunk in generar_diagnostico_stream(datos_empresa):
        print(chunk, end="", flush=True)
        full_text += chunk

    print("\n\n" + "=" * 50)
    print("✅ Diagnóstico completado exitosamente.")
    return full_text


if __name__ == "__main__":
    # Prueba de ejecución directa desde la terminal
    csv_file = os.path.join(PROJECT_ROOT, "data", "template_evaluacion.csv")
    
    if not os.path.exists(csv_file):
        csv_file = "data/template_evaluacion.csv"

    if not os.path.exists(csv_file):
        print(f"⚠️ El archivo '{csv_file}' no existe. Verifica la ruta en la carpeta 'data'.")
    else:
        df = pd.read_csv(csv_file)
        datos_empresa = df.to_string(index=False)
        generar_diagnostico_cli(datos_empresa)