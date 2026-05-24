import streamlit as st
import joblib
import pandas as pd
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ===== SETUP NLTK =====
nltk.download('stopwords', quiet=True)

# ===== CONFIGURAÇÃO DA PÁGINA =====
st.set_page_config(
    page_title="SentimentLab",
    page_icon="🎬",
    layout="wide"
)

# ===== CONSTANTES =====
MODELS_DIR = "models/"

MODELOS_DISPONIVEIS = {
    "Regressão Logística": "Regressao_Logistica",
    "SVM": "SVM",
    "Naive Bayes": "Naive_Bayes",
    "Random Forest": "Random_Forest"
}

VETORIZACOES = ["TF-IDF", "BoW"]

# ===== PRÉ-PROCESSAMENTO =====
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r"'", ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)

# ===== CARREGAR MODELOS =====
@st.cache_resource
def carregar_modelo(modelo_nome, vet_nome):
    modelo_path = os.path.join(MODELS_DIR, f"{modelo_nome}_{vet_nome}.pkl")
    vet_path = os.path.join(MODELS_DIR, f"vectorizer_{vet_nome}.pkl")
    modelo = joblib.load(modelo_path)
    vectorizer = joblib.load(vet_path)
    return modelo, vectorizer

# ===== NAVEGAÇÃO =====
st.sidebar.title("🎬 SentimentLab")
st.sidebar.markdown("Análise comparativa de algoritmos de Machine Learning para classificação de sentimentos.")

st.sidebar.divider()

st.sidebar.markdown("""
**📚 Projeto Acadêmico**  
Trabalho de Conclusão de Curso  
Engenharia de Software — UNINTER  

**Desenvolvido por**  
Janyelle Oliveira & Júlia Oliveira 

**📅 2026**
""")

st.sidebar.divider()

pagina = st.sidebar.radio("Navegação", ["🔍 Analisar Texto", "📊 Relatório de Desempenho"])
# =====================
# TELA 1 — ANALISAR TEXTO
# =====================
if pagina == "🔍 Analisar Texto":
    st.title("🔍 Analisar Texto")
    st.markdown("Digite uma avaliação de filme e veja a predição de sentimento dos modelos.")

    texto = st.text_area("📝 Digite sua avaliação:", height=150,
                         placeholder="Ex: This movie was absolutely amazing! The acting was superb...")

    col1, col2 = st.columns(2)
    with col1:
        modelo_selecionado = st.selectbox("🤖 Modelo", list(MODELOS_DISPONIVEIS.keys()) + ["Comparar todos"])
    with col2:
        vet_selecionado = st.selectbox("📐 Vetorização", VETORIZACOES)

    if st.button("Analisar", type="primary"):
        if not texto.strip():
            st.warning("Por favor, digite uma avaliação antes de analisar.")
        else:
            texto_processado = preprocess(texto)

            if modelo_selecionado == "Comparar todos":
                st.subheader("Resultado — Comparação de todos os modelos")
                resultados = []
                for nome_display, nome_arquivo in MODELOS_DISPONIVEIS.items():
                    modelo, vectorizer = carregar_modelo(nome_arquivo, vet_selecionado)
                    X = vectorizer.transform([texto_processado])
                    pred = modelo.predict(X)[0]
                    label = "😊 Positivo" if pred == 1 else "😞 Negativo"

                    # Confiança (apenas para modelos que suportam predict_proba)
                    confianca = "—"
                    if hasattr(modelo, "predict_proba"):
                        prob = modelo.predict_proba(X)[0]
                        confianca = f"{max(prob)*100:.1f}%"

                    resultados.append({
                        "Modelo": nome_display,
                        "Resultado": label,
                        "Confiança": confianca
                    })

                st.dataframe(pd.DataFrame(resultados), use_container_width=True, hide_index=True)

            else:
                nome_arquivo = MODELOS_DISPONIVEIS[modelo_selecionado]
                modelo, vectorizer = carregar_modelo(nome_arquivo, vet_selecionado)
                X = vectorizer.transform([texto_processado])
                pred = modelo.predict(X)[0]

                if pred == 1:
                    st.success("## 😊 POSITIVO")
                else:
                    st.error("## 😞 NEGATIVO")

                if hasattr(modelo, "predict_proba"):
                    prob = modelo.predict_proba(X)[0]
                    confianca = max(prob) * 100
                    st.metric("Confiança", f"{confianca:.1f}%")

# =====================
# TELA 2 — RELATÓRIO
# =====================
elif pagina == "📊 Relatório de Desempenho":
    st.title("📊 Relatório de Desempenho")
    st.markdown("Comparação das métricas dos 8 experimentos (4 modelos × 2 vetorizações).")

    try:
        df = pd.read_csv("results/resultados.csv")

        # Destaque do melhor modelo
        melhor = df.loc[df['F1-Score'].idxmax()]
        st.success(f"🏆 Melhor modelo: **{melhor['Modelo']} + {melhor['Vetorização']}** — F1-Score: {melhor['F1-Score']}")

        # Tabela completa
        st.subheader("Tabela Comparativa")
        st.dataframe(df.sort_values('F1-Score', ascending=False), use_container_width=True, hide_index=True)

        # Gráficos
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Comparação de F1-Score")
            st.image("results/comparacao_f1.png")
        with col2:
            st.subheader("Matriz de Confusão — Melhor Modelo")
            st.image("results/matriz_confusao.png")

    except FileNotFoundError:
        st.error("Arquivo results/resultados.csv não encontrado. Execute o notebook 03 primeiro.")

# ===== RODAPÉ =====
st.divider()
st.caption("SentimentLab — TCC Engenharia de Software · UNINTER · Janyelle Oliveira & Júlia Oliveira · 2026")