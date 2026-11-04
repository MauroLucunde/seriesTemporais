"""
G11 — Demonstração didáctica de ECG sintético e Machine Learning
Interface gráfica desenvolvida com Streamlit.

AVISO: dados e rótulos artificiais. Este projecto não é um dispositivo médico
e não deve ser utilizado para diagnóstico ou decisões clínicas.
"""

import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

# ---------------------------------------------------------------------
# CONFIGURAÇÕES GERAIS
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="G11 | Análise didáctica de ECG",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

FS = 250
DURACAO = 8
NOMES_CLASSES = ["Padrão A", "Padrão B"]
COLUNAS_CARACTERISTICAS = [
    "Média", "Desvio padrão", "Amplitude pico a pico",
    "RMS", "Energia média"
]


# ---------------------------------------------------------------------
# GERAÇÃO E PROCESSAMENTO DOS SINAIS
# ---------------------------------------------------------------------
def gerar_sinal_sintetico(rotulo: int, rng: np.random.Generator,
                          fs: int = FS, duracao: int = DURACAO):
    """Cria um sinal ECG artificial para demonstração académica."""
    t = np.arange(fs * duracao) / fs
    hr = rng.uniform(58, 92) if rotulo == 0 else rng.uniform(85, 135)
    intervalo = 60.0 / hr

    batimentos = []
    instante = rng.uniform(0.2, 0.5)

    while instante < duracao - 0.25:
        batimentos.append(instante)
        if rotulo == 0:
            instante += intervalo + rng.normal(0, 0.025)
        else:
            instante += intervalo * rng.uniform(0.70, 1.35) + rng.normal(0, 0.035)

    sinal = np.zeros_like(t)

    for b in batimentos:
        sinal += 0.12 * np.exp(-0.5 * ((t - (b - 0.18)) / 0.035) ** 2)
        sinal += -0.15 * np.exp(-0.5 * ((t - (b - 0.025)) / 0.010) ** 2)

        amplitude = 1.0 if rotulo == 0 else rng.uniform(0.65, 1.35)
        sinal += amplitude * np.exp(-0.5 * ((t - b) / 0.014) ** 2)
        sinal += -0.25 * np.exp(-0.5 * ((t - (b + 0.025)) / 0.012) ** 2)
        sinal += 0.28 * np.exp(-0.5 * ((t - (b + 0.25)) / 0.060) ** 2)

    sinal += 0.08 * np.sin(2 * np.pi * 0.25 * t + rng.uniform(0, 2 * np.pi))
    sinal += 0.025 * np.sin(2 * np.pi * 50 * t + rng.uniform(0, 2 * np.pi))
    ruido = 0.035 if rotulo == 0 else 0.055
    sinal += rng.normal(0, ruido, size=len(t))

    return t, sinal


def filtrar_ecg(sinal: np.ndarray, fs: int = FS):
    """Aplica filtro Butterworth passa-banda de 0,5 a 40 Hz."""
    sos = butter(4, [0.5, 40], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, sinal)


def extrair_caracteristicas(sinal: np.ndarray):
    """Extrai características estatísticas simples do sinal."""
    return {
        "Média": float(np.mean(sinal)),
        "Desvio padrão": float(np.std(sinal)),
        "Amplitude pico a pico": float(np.ptp(sinal)),
        "RMS": float(np.sqrt(np.mean(sinal ** 2))),
        "Energia média": float(np.mean(sinal ** 2)),
    }


@st.cache_data(show_spinner=False)
def criar_dataset(n_pessoas: int, janelas_por_pessoa: int, semente: int):
    """Gera o conjunto de dados e mantém o identificador de cada pessoa."""
    rng = np.random.default_rng(semente)
    registos = []
    exemplo = None

    for pessoa in range(n_pessoas):
        rotulo = pessoa % 2

        for janela in range(janelas_por_pessoa):
            t, bruto = gerar_sinal_sintetico(rotulo, rng)
            filtrado = filtrar_ecg(bruto)

            linha = extrair_caracteristicas(filtrado)
            linha["classe"] = rotulo
            linha["pessoa"] = f"P{pessoa + 1:03d}"
            linha["janela"] = janela + 1
            registos.append(linha)

            if pessoa == 0 and janela == 0:
                exemplo = (t, bruto, filtrado)

    return pd.DataFrame(registos), exemplo


# ---------------------------------------------------------------------
# TREINO E AVALIAÇÃO
# ---------------------------------------------------------------------
def treinar_avaliar(df: pd.DataFrame, n_estimators: int,
                    test_size: float, semente: int):
    """Divide por pessoa, treina o modelo e calcula as métricas."""
    X = df[COLUNAS_CARACTERISTICAS]
    y = df["classe"].to_numpy()
    grupos = df["pessoa"].to_numpy()

    separador = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=semente
    )
    idx_treino, idx_teste = next(separador.split(X, y, groups=grupos))

    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=semente,
        n_jobs=-1,
    )
    modelo.fit(X.iloc[idx_treino], y[idx_treino])

    previsoes = modelo.predict(X.iloc[idx_teste])
    probabilidades = modelo.predict_proba(X.iloc[idx_teste])[:, 1]
    reais = y[idx_teste]

    cm = confusion_matrix(reais, previsoes, labels=[0, 1])
    metricas = {
        "Acurácia": accuracy_score(reais, previsoes),
        "Precisão (Padrão B)": precision_score(reais, previsoes, zero_division=0),
        "Sensibilidade (Padrão B)": recall_score(reais, previsoes, zero_division=0),
        "F1-score (Padrão B)": f1_score(reais, previsoes, zero_division=0),
        "AUC-ROC": (
            roc_auc_score(reais, probabilidades)
            if len(np.unique(reais)) == 2 else float("nan")
        ),
    }

    previsoes_df = pd.DataFrame({
        "Pessoa": grupos[idx_teste],
        "Classe real": reais,
        "Classe prevista": previsoes,
        "Probabilidade de Padrão B": probabilidades,
    })

    relatorio = classification_report(
        reais, previsoes, labels=[0, 1], target_names=NOMES_CLASSES,
        zero_division=0
    )

    return modelo, metricas, cm, previsoes_df, relatorio, idx_treino, idx_teste


# ---------------------------------------------------------------------
# INTERFACE STREAMLIT
# ---------------------------------------------------------------------
st.title("❤️ G11 — Análise de sinais ECG com Machine Learning")
st.caption("Aplicação didáctica para geração, filtragem, análise e classificação de sinais ECG sintéticos.")

st.warning(
    "Demonstração exclusivamente académica. Os sinais e rótulos são artificiais; "
    "os resultados não têm validade clínica e não permitem diagnosticar doenças."
)

with st.sidebar:
    st.header("⚙️ Configurações")
    st.write("Personalize a experiência e execute a simulação.")

    n_pessoas = st.slider("Número de pessoas sintéticas", 20, 200, 60, step=10)
    janelas = st.slider("Janelas por pessoa", 2, 10, 4)
    n_estimators = st.slider("Árvores do Random Forest", 50, 500, 150, step=50)
    test_size = st.slider("Proporção para teste", 0.15, 0.40, 0.25, step=0.05)
    semente = st.number_input(
        "Semente aleatória",
        min_value=1,
        max_value=50_000_000,
        value=20261026,
        step=1,
    )

    executar = st.button(
        "🚀 Gerar dados e treinar modelo",
        type="primary",
        use_container_width=True,
    )

if "executado" not in st.session_state:
    st.session_state.executado = False

if executar:
    with st.spinner("A gerar sinais, a filtrar e a treinar o modelo..."):
        dados, exemplo = criar_dataset(n_pessoas, janelas, int(semente))
        resultado = treinar_avaliar(dados, n_estimators, test_size, int(semente))

        st.session_state.dados = dados
        st.session_state.exemplo = exemplo
        st.session_state.resultado = resultado
        st.session_state.executado = True

if not st.session_state.executado:
    st.info("Ajuste os parâmetros na barra lateral e clique em **Gerar dados e treinar modelo**.")
    st.subheader("O que esta aplicação faz?")
    c1, c2, c3 = st.columns(3)
    c1.markdown("### 1. Geração\nCria sinais ECG artificiais de duas classes.")
    c2.markdown("### 2. Pré-processamento\nAplica filtro passa-banda para reduzir componentes de ruído.")
    c3.markdown("### 3. Machine Learning\nTreina e avalia um classificador Random Forest.")
    st.stop()

df = st.session_state.dados
t, bruto, filtrado = st.session_state.exemplo
modelo, metricas, cm, previsoes_df, relatorio, idx_treino, idx_teste = st.session_state.resultado

# Navegação por separadores
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Visão geral", "〰️ Sinal ECG", "🧠 Machine Learning",
    "📋 Dados e previsões", "⬇️ Exportar resultados"
])

with tab1:
    st.subheader("Resumo da simulação")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pessoas sintéticas", f"{df['pessoa'].nunique()}")
    m2.metric("Total de janelas", f"{len(df)}")
    m3.metric("Janelas de treino", f"{len(idx_treino)}")
    m4.metric("Janelas de teste", f"{len(idx_teste)}")

    st.markdown("#### Distribuição das classes")
    distribuicao = (
        df["classe"]
        .map({0: "Padrão A", 1: "Padrão B"})
        .value_counts()
        .reindex(NOMES_CLASSES, fill_value=0)
    )
    st.bar_chart(distribuicao)

    st.markdown("#### Métricas do modelo")
    cols = st.columns(len(metricas))
    for coluna, (nome, valor) in zip(cols, metricas.items()):
        coluna.metric(nome, "N/D" if np.isnan(valor) else f"{valor:.3f}")

with tab2:
    st.subheader("Visualização do sinal ECG")
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(t, bruto, label="Sinal bruto", linewidth=0.8, alpha=0.75)
    ax.plot(t, filtrado, label="Sinal filtrado", linewidth=1.0)
    ax.set_xlabel("Tempo (segundos)")
    ax.set_ylabel("Amplitude (unidade arbitrária)")
    ax.set_title("Comparação antes e depois do filtro passa-banda")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Parâmetros do filtro**")
    st.write("- Tipo: Butterworth passa-banda")
    st.write("- Frequências de corte: 0,5 Hz e 40 Hz")
    st.write(f"- Frequência de amostragem: {FS} Hz")

with tab3:
    st.subheader("Avaliação do classificador Random Forest")
    esquerda, direita = st.columns(2)

    with esquerda:
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=NOMES_CLASSES
        )
        disp.plot(ax=ax, values_format="d", colorbar=False)
        ax.set_title("Matriz de confusão")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with direita:
        importancia = pd.Series(
            modelo.feature_importances_,
            index=COLUNAS_CARACTERISTICAS
        ).sort_values()
        st.markdown("**Importância das características**")
        st.bar_chart(importancia)

    st.markdown("#### Relatório de classificação")
    st.code(relatorio, language="text")

    st.caption(
        "A separação treino/teste é feita por pessoa (GroupShuffleSplit), "
        "evitando que janelas da mesma pessoa apareçam nos dois conjuntos."
    )

with tab4:
    st.subheader("Amostra dos dados gerados")
    st.dataframe(df.head(20), use_container_width=True)
    st.subheader("Previsões no conjunto de teste")
    st.dataframe(previsoes_df, use_container_width=True)

with tab5:
    st.subheader("Descarregar resultados")
    metricas_exportacao = {
        "total_janelas": int(len(df)),
        "total_pessoas_sinteticas": int(df["pessoa"].nunique()),
        "janelas_treino": int(len(idx_treino)),
        "janelas_teste": int(len(idx_teste)),
        "metricas": {k: (None if np.isnan(v) else float(v)) for k, v in metricas.items()},
        "matriz_confusao_linhas_real_colunas_previsto": cm.tolist(),
        "aviso": "Dados e rótulos sintéticos; métricas sem significado clínico."
    }

    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "⬇️ Dados completos (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="dados_ecg_sinteticos.csv",
        mime="text/csv",
        use_container_width=True,
    )
    c2.download_button(
        "⬇️ Previsões (CSV)",
        data=previsoes_df.to_csv(index=False).encode("utf-8"),
        file_name="previsoes_teste.csv",
        mime="text/csv",
        use_container_width=True,
    )
    c3.download_button(
        "⬇️ Métricas (JSON)",
        data=json.dumps(metricas_exportacao, ensure_ascii=False, indent=2),
        file_name="metricas.json",
        mime="application/json",
        use_container_width=True,
    )

    st.download_button(
        "⬇️ Relatório de classificação (TXT)",
        data=relatorio,
        file_name="relatorio_classificacao.txt",
        mime="text/plain",
    )

st.divider()
st.caption("Projecto G11 | Demonstração académica de processamento de sinais ECG e aprendizagem automática.")
