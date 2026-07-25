# 🌿 EcoCircular - Agente IA de Cumplimiento Normativo (RAG)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-f34f29?style=flat)
![OCI](https://img.shields.io/badge/Oracle_Cloud-OCI_Infrastructure-F80000?style=flat&logo=oracle&logoColor=white)

## 📌 Descripción General del Proyecto

**EcoCircular AgentIA** es un agente de Inteligencia Artificial fundado en arquitectura **RAG (Retrieval-Augmented Generation)**. Su objetivo principal es auditar, analizar y verificar el cumplimiento de normativas de sostenibilidad, gestión ambiental y economía circular a partir de documentos de consulta en formato PDF o CSV.

El sistema permite a auditores y consultores realizar preguntas en lenguaje natural sobre la documentación cargada, entregando respuestas precisas y basadas estrictamente en la evidencia del texto original.

---

## 🏗️ Arquitectura de la Solución

El flujo del agente inteligente se estructura de la siguiente manera:

1. **Interfaz de Usuario (Streamlit):** Permite la carga interactiva de documentos (PDF/CSV) y la formulación de consultas.
2. **Procesamiento de Documentos:** Extracción y segmentación (*chunking*) del texto del archivo fuente.
3. **Indexación y Búsqueda Semántica (ChromaDB):** Generación de embeddings vectoriales para recuperar los fragmentos normativos más relevantes a la consulta del usuario.
4. **Motor de Inferencia LLM (Groq API):** Generación de respuestas utilizando el modelo **Llama 3.3 70B Versatile** condicionado por el contexto recuperado.
5. **Infraestructura Cloud (OCI):** Servidor Compute alojado en Oracle Cloud Infrastructure dentro de una VCN con subred pública.

---

## 🛠️ Tecnologías y Herramientas Utilizadas

* **Lenguaje:** Python 3.10+
* **Framework Web:** Streamlit
* **Modelo LLM:** Llama 3.3 70B (vía Groq Cloud API)
* **Base de Datos Vectorial:** ChromaDB / LangChain Embeddings
* **Procesamiento de Documentos:** PyPDF2 / Pandas
* **Infraestructura Cloud:** Oracle Cloud Infrastructure (OCI) - Instancia Compute (`VM.Standard.A1.Flex`) en la región de México Central (Querétaro).

---

## 💻 Instrucciones para Ejecutar el Proyecto Localmente

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/TU_USUARIO/ecocircular_agentIA_Alura.git](https://github.com/TU_USUARIO/ecocircular_agentIA_Alura.git)
   cd ecocircular_agentIA_Alura
