import streamlit as st
import pandas as pd
from src.agent import generar_diagnostico_stream

st.set_page_config(
    page_title="EcoCircular AI - Auditoría de Economía Circular",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 EcoCircular AI")
st.caption("Plataforma de Auditoría de Cumplimiento de la Ley General de Economía Circular para MiPyMEs en México")

st.markdown("---")

# Sidebar para carga de archivos
st.sidebar.header("📂 Carga de Datos")
uploaded_file = st.sidebar.file_uploader(
    "Sube el archivo CSV con los datos operativos", 
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    datos_empresa = df.to_string(index=False)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("📊 Datos Operativos Cargados")
        st.dataframe(df, use_container_width=True)

    with col_right:
        st.subheader("📋 Informe de Auditoría")
        
        if st.button("🚀 Generar Diagnóstico", type="primary"):
            reporte_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("Consultando marco legal en ChromaDB y analizando con Llama 3.3..."):
                for chunk in generar_diagnostico_stream(datos_empresa):
                    full_response += chunk
                    reporte_placeholder.markdown(full_response + "▌")
                
                reporte_placeholder.markdown(full_response)
                
            st.success("✅ Diagnóstico completado exitosamente.")
            
            # Botón para descargar el reporte generado
            st.download_button(
                label="📥 Descargar Reporte (.md)",
                data=full_response,
                file_name="diagnostico_ecocircular.md",
                mime="text/markdown"
            )
else:
    st.info("👈 Por favor, sube un archivo CSV desde el panel lateral para iniciar la auditoría.")