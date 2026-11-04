"""
T11 — Análise de séries temporais biomédicas (ECG sintético)
Unidade IV — Algoritmos de Diagnóstico e Apoio à Decisão

AVISO: dados e rótulos artificiais. Este projecto não é um dispositivo médico
e não deve ser utilizado para diagnóstico ou decisões clínicas.
"""

import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.signal import (
    butter, sosfiltfilt, sosfreqz, iirnotch, filtfilt,
    find_peaks, welch, spectrogram,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, classification_report,
    ConfusionMatrixDisplay,
)

# ---------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="T11 | Séries temporais biomédicas",
    page_icon="❤️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem;}
    .stTabs [data-baseweb="tab"] {padding: 6px 12px;}
    div[data-testid="stMetric"] {
        background: rgba(128,128,128,.08);
        border: 1px solid rgba(128,128,128,.22);
        padding: 10px 14px; border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

NOMES_CLASSES = ["Padrão A", "Padrão B"]
NORMALIZACOES = [
    "Nenhuma", "Z-score por janela", "Min-Max por janela",
    "Z-score por registo (doente)",
]
GRUPOS_FEAT = ["Estatísticas temporais", "Ritmo (picos R)", "Espectrais"]

# Valores por omissão de TODOS os parâmetros (cada um vive na sua aba)
DEFAULTS = dict(
    n_pessoas=60, prevalencia=0.5, variab_ganho=0.30, semente=20261026,
    pessoa_exemplo=1,
    fs=250, duracao=30,
    ruido=0.035, deriva=0.08, rede=0.025, n_art=2, amp_art=0.5,
    low=0.5, high=40, ordem=4, notch=True,
    janela_s=4.0, sobreposicao=0.0, normalizacao="Z-score por registo (doente)",
    nperseg=128,
    grupos_feat=list(GRUPOS_FEAT),
    prop_teste=0.20, prop_val=0.20, mostrar_fuga=False,
    n_estimators=150, max_depth=0, class_weight="balanced",
    limiar=0.50, usar_limiar_val=False, sens_alvo=0.90, custo_fn=5,
    n_boot=300,
)


def P(chave):
    """Lê o valor actual de um parâmetro (ou o valor por omissão)."""
    return st.session_state.get(chave, DEFAULTS[chave])


def repor_parametros():
    for k in DEFAULTS:
        st.session_state.pop(k, None)


def fmt(v, casas=3):
    return "N/D" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{casas}f}"


# ---------------------------------------------------------------------
# GERAÇÃO DO SINAL (amostragem, ruído e artefactos)
# ---------------------------------------------------------------------
def gerar_registo(rotulo, rng, fs, duracao, ruido, deriva, rede,
                  n_art, amp_art, ganho):
    """ECG artificial de um doente: ritmo, ruído, deriva, 50 Hz e artefactos."""
    t = np.arange(int(fs * duracao)) / fs
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

    ecg = np.zeros_like(t)
    for b in batimentos:
        ecg += 0.12 * np.exp(-0.5 * ((t - (b - 0.18)) / 0.035) ** 2)      # onda P
        ecg += -0.15 * np.exp(-0.5 * ((t - (b - 0.025)) / 0.010) ** 2)    # Q
        amplitude = 1.0 if rotulo == 0 else rng.uniform(0.65, 1.35)
        ecg += amplitude * np.exp(-0.5 * ((t - b) / 0.014) ** 2)          # R
        ecg += -0.25 * np.exp(-0.5 * ((t - (b + 0.025)) / 0.012) ** 2)   # S
        ecg += 0.28 * np.exp(-0.5 * ((t - (b + 0.25)) / 0.060) ** 2)      # T

    sinal = ganho * ecg
    sinal += deriva * np.sin(2 * np.pi * 0.25 * t + rng.uniform(0, 2 * np.pi))
    if rede > 0 and fs > 100:
        sinal += rede * np.sin(2 * np.pi * 50 * t + rng.uniform(0, 2 * np.pi))
    nivel = ruido if rotulo == 0 else ruido * 1.5
    sinal += rng.normal(0, nivel, size=len(t))

    for _ in range(int(n_art)):  # artefactos de movimento
        n = int(rng.uniform(0.3, 0.8) * fs)
        i = int(rng.integers(0, len(t) - n))
        tau = np.arange(n) / fs
        f_art = rng.uniform(1, 5)
        sinal[i:i + n] += (amp_art * np.hanning(n)
                           * np.sin(2 * np.pi * f_art * tau + rng.uniform(0, 2 * np.pi)))
    return t, sinal


# ---------------------------------------------------------------------
# PRÉ-PROCESSAMENTO
# ---------------------------------------------------------------------
def filtrar_ecg(sinal, fs, low, high, ordem, notch):
    nyq = fs / 2
    high = min(high, 0.95 * nyq)
    sos = butter(ordem, [low, high], btype="bandpass", fs=fs, output="sos")
    y = sosfiltfilt(sos, sinal)
    if notch and 50 < 0.98 * nyq:
        b, a = iirnotch(50, 30, fs=fs)
        y = filtfilt(b, a, y)
    return y


def normalizar_registo(x, modo):
    if modo == "Z-score por registo (doente)":
        return (x - x.mean()) / (x.std() + 1e-12)
    return x


def normalizar_janela(x, modo):
    if modo == "Z-score por janela":
        return (x - x.mean()) / (x.std() + 1e-12)
    if modo == "Min-Max por janela":
        return (x - x.min()) / (np.ptp(x) + 1e-12)
    return x


def segmentar(x, fs, janela_s, sobreposicao):
    n = int(round(janela_s * fs))
    passo = max(1, int(round(n * (1 - sobreposicao))))
    return [x[i:i + n] for i in range(0, len(x) - n + 1, passo)]


# ---------------------------------------------------------------------
# CARACTERÍSTICAS
# ---------------------------------------------------------------------
def extrair_caracteristicas(x, fs, grupos):
    f = {}
    if "Estatísticas temporais" in grupos:
        f["Média"] = float(np.mean(x))
        f["Desvio padrão"] = float(np.std(x))
        f["Amplitude pico a pico"] = float(np.ptp(x))
        f["RMS"] = float(np.sqrt(np.mean(x ** 2)))
        f["Energia média"] = float(np.mean(x ** 2))
    if "Ritmo (picos R)" in grupos:
        amp = np.ptp(x)
        picos, _ = find_peaks(x, prominence=0.35 * amp,
                              distance=max(1, int(0.25 * fs)))
        rr = np.diff(picos) / fs
        f["Nº de picos R"] = float(len(picos))
        f["FC estimada (bpm)"] = float(60 / np.mean(rr)) if len(rr) > 0 else 0.0
        f["SDRR (s)"] = float(np.std(rr)) if len(rr) > 1 else 0.0
        f["RMSSD (s)"] = float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) > 2 else 0.0
    if "Espectrais" in grupos:
        freqs, psd = welch(x, fs=fs, nperseg=min(len(x), 256))
        total = np.sum(psd) + 1e-12

        def banda(a, b):
            m = (freqs >= a) & (freqs < b)
            return float(np.sum(psd[m]) / total)

        p = psd / total
        f["Potência rel. 0,5–5 Hz"] = banda(0.5, 5)
        f["Potência rel. 5–15 Hz"] = banda(5, 15)
        f["Potência rel. 15–40 Hz"] = banda(15, 40)
        f["Frequência dominante (Hz)"] = float(freqs[np.argmax(psd)])
        f["Entropia espectral"] = float(-np.sum(p * np.log2(p + 1e-12)) / np.log2(len(p)))
    return f


@st.cache_data(show_spinner="A gerar e processar os sinais...")
def criar_dataset(n_pessoas, prevalencia, duracao, fs, semente, ruido, deriva,
                  rede, n_art, amp_art, variab_ganho, low, high, ordem, notch,
                  janela_s, sobreposicao, normalizacao, grupos):
    rng = np.random.default_rng(semente)
    n_b = int(np.clip(round(n_pessoas * prevalencia), 4, n_pessoas - 4))
    rotulos = np.array([1] * n_b + [0] * (n_pessoas - n_b))
    rng.shuffle(rotulos)

    registos, exemplos = [], {}
    contagem = {0: 0, 1: 0}

    for i, rotulo in enumerate(rotulos):
        pid = f"P{i + 1:03d}"
        ganho = float(rng.lognormal(0, variab_ganho))
        t, bruto = gerar_registo(int(rotulo), rng, fs, duracao, ruido, deriva,
                                 rede, n_art, amp_art, ganho)
        filtrado = filtrar_ecg(bruto, fs, low, high, ordem, notch)
        registo_norm = normalizar_registo(filtrado, normalizacao)

        if contagem[int(rotulo)] < 4:  # guarda 4 exemplos de cada classe
            exemplos[pid] = dict(t=t, bruto=bruto, filtrado=filtrado,
                                 classe=int(rotulo), ganho=ganho)
            contagem[int(rotulo)] += 1

        for j, janela in enumerate(segmentar(registo_norm, fs, janela_s, sobreposicao)):
            x = normalizar_janela(janela, normalizacao)
            linha = extrair_caracteristicas(x, fs, grupos)
            linha.update(classe=int(rotulo), pessoa=pid, janela=j + 1)
            registos.append(linha)

    return pd.DataFrame(registos), dict(sorted(exemplos.items()))


# ---------------------------------------------------------------------
# DIVISÃO POR DOENTE, TREINO E MÉTRICAS
# ---------------------------------------------------------------------
def dividir_por_doente(df, prop_teste, prop_val, semente):
    """Divide DOENTES (não janelas) em treino/validação/teste, estratificado."""
    pessoas = df.groupby("pessoa")["classe"].first()
    ids, y = pessoas.index.to_numpy(), pessoas.to_numpy()
    ids_trv, ids_te, y_trv, _ = train_test_split(
        ids, y, test_size=prop_teste, stratify=y, random_state=semente)
    ids_tr, ids_va = train_test_split(
        ids_trv, test_size=prop_val / (1 - prop_teste),
        stratify=y_trv, random_state=semente)
    pes = df["pessoa"].to_numpy()
    return (np.where(np.isin(pes, ids_tr))[0],
            np.where(np.isin(pes, ids_va))[0],
            np.where(np.isin(pes, ids_te))[0])


def criar_modelo(n_estimators, max_depth, class_weight, semente):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None if max_depth == 0 else max_depth,
        class_weight="balanced" if class_weight == "balanced" else None,
        random_state=semente, n_jobs=-1,
    )


@st.cache_data(show_spinner="A treinar o modelo...")
def treinar(df, colunas, idx_tr, idx_va, idx_te,
            n_estimators, max_depth, class_weight, semente):
    X = df[list(colunas)].to_numpy()
    y = df["classe"].to_numpy()
    modelo = criar_modelo(n_estimators, max_depth, class_weight, semente)
    modelo.fit(X[idx_tr], y[idx_tr])

    def prever(idx):
        p = modelo.predict_proba(X[idx])[:, 1]
        arvores = np.array([a.predict_proba(X[idx])[:, 1] for a in modelo.estimators_])
        return p, arvores.std(axis=0)

    p_tr, _ = prever(idx_tr)
    p_va, s_va = prever(idx_va)
    p_te, s_te = prever(idx_te)
    return dict(modelo=modelo, p_tr=p_tr, p_va=p_va, s_va=s_va,
                p_te=p_te, s_te=s_te,
                importancia=modelo.feature_importances_)


@st.cache_data(show_spinner=False)
def comparar_divisao_errada(X, y, n_estimators, max_depth, class_weight,
                            prop_teste, semente):
    """Divisão ERRADA por janela, para mostrar o efeito da fuga de dados."""
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=prop_teste, stratify=y, random_state=semente)
    m = criar_modelo(n_estimators, max_depth, class_weight, semente)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    return float(np.mean((p >= 0.5) == yte)), float(roc_auc_score(yte, p))


def calcular_metricas(y, p, limiar):
    pred = (p >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    div = lambda a, b: a / b if b else float("nan")
    sens, esp = div(tp, tp + fn), div(tn, tn + fp)
    prec = div(tp, tp + fp)
    f1 = div(2 * prec * sens, prec + sens) if not np.isnan(prec) and not np.isnan(sens) else float("nan")
    dois = len(np.unique(y)) == 2
    metricas = {
        "Acurácia": (tp + tn) / len(y),
        "Acurácia balanceada": float(np.nanmean([sens, esp])),
        "Sensibilidade": sens,
        "Especificidade": esp,
        "Precisão (VPP)": prec,
        "F1-score": f1,
        "AUC-ROC": float(roc_auc_score(y, p)) if dois else float("nan"),
        "AUC-PR": float(average_precision_score(y, p)) if dois else float("nan"),
    }
    return metricas, (int(tn), int(fp), int(fn), int(tp))


def limiar_para_sensibilidade(y, p, alvo):
    pos = p[y == 1]
    if len(pos) == 0:
        return 0.5
    return float(np.clip(np.quantile(pos, 1 - alvo), 0.01, 0.99))


@st.cache_data(show_spinner="A calcular intervalos de confiança...")
def ic_bootstrap(y, p, pessoas, limiar, n_boot, semente):
    """Bootstrap por DOENTE (reamostra pessoas, não janelas)."""
    rng = np.random.default_rng(semente)
    ids = np.unique(pessoas)
    grupos = {i: np.where(pessoas == i)[0] for i in ids}
    chaves = ["Sensibilidade", "Especificidade", "AUC-ROC", "Acurácia balanceada"]
    res = {k: [] for k in chaves}
    for _ in range(n_boot):
        amostra = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([grupos[i] for i in amostra])
        if len(np.unique(y[idx])) < 2:
            continue
        m, _ = calcular_metricas(y[idx], p[idx], limiar)
        for k in chaves:
            res[k].append(m[k])
    out = {}
    for k, v in res.items():
        v = np.array(v, dtype=float)
        out[k] = ((float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
                  if len(v) > 10 and not np.all(np.isnan(v)) else (float("nan"), float("nan")))
    return out


# ---------------------------------------------------------------------
# CABEÇALHO
# ---------------------------------------------------------------------
c_tit, c_btn = st.columns([5, 1])
c_tit.title(" Análise de séries temporais biomédicas")
c_tit.caption("Unidade IV · Algoritmos de Diagnóstico e Apoio à Decisão · "
              "ECG sintético com pré-processamento e aprendizagem automática")
c_btn.write("")
c_btn.button("↩️ Repor parâmetros", on_click=repor_parametros)

st.warning(
    "Demonstração exclusivamente académica. Os sinais e os rótulos são artificiais; "
    "os resultados não têm validade clínica e não permitem diagnosticar doenças."
)

# ---------------------------------------------------------------------
# PIPELINE (lê os parâmetros definidos nas abas)
# ---------------------------------------------------------------------
grupos_sel = tuple(P("grupos_feat")) or (GRUPOS_FEAT[0],)
semente = int(P("semente"))

df, exemplos = criar_dataset(
    int(P("n_pessoas")), float(P("prevalencia")), int(P("duracao")), int(P("fs")),
    semente, float(P("ruido")), float(P("deriva")), float(P("rede")),
    int(P("n_art")), float(P("amp_art")), float(P("variab_ganho")),
    float(P("low")), float(P("high")), int(P("ordem")), bool(P("notch")),
    float(P("janela_s")), float(P("sobreposicao")), P("normalizacao"), grupos_sel,
)
colunas = [c for c in df.columns if c not in ("classe", "pessoa", "janela")]
y_all = df["classe"].to_numpy()

try:
    idx_tr, idx_va, idx_te = dividir_por_doente(
        df, float(P("prop_teste")), float(P("prop_val")), semente)
except ValueError as erro:
    st.error(f"Não foi possível dividir os doentes por classe: {erro}. "
             "Aumente o número de pessoas ou a prevalência do Padrão B.")
    st.stop()

R = treinar(df, tuple(colunas), idx_tr, idx_va, idx_te,
            int(P("n_estimators")), int(P("max_depth")), P("class_weight"), semente)
y_tr, y_va, y_te = y_all[idx_tr], y_all[idx_va], y_all[idx_te]
pes_te = df["pessoa"].to_numpy()[idx_te]

limiar = float(P("limiar"))
if P("usar_limiar_val"):
    limiar = limiar_para_sensibilidade(y_va, R["p_va"], float(P("sens_alvo")))

met_te, (tn, fp, fn, tp) = calcular_metricas(y_te, R["p_te"], limiar)
met_va, _ = calcular_metricas(y_va, R["p_va"], limiar)

ids_ex = list(exemplos.keys())
pid_ex = ids_ex[int(P("pessoa_exemplo")) - 1]
ex = exemplos[pid_ex]
fs = int(P("fs"))

# ---------------------------------------------------------------------
# ABAS
# ---------------------------------------------------------------------
(t_resumo, t_coorte, t_amostr, t_ruido, t_filtro, t_seg, t_tf, t_feat,
 t_div, t_modelo, t_metr, t_inc, t_exp) = st.tabs([
    "📋 Resumo", "👥 Coorte e classes", "📡 Amostragem", "🔊 Ruído e artefactos",
    "🎛️ Filtragem", "✂️ Segmentação e normalização", "〰️ Tempo-frequência",
    "🧮 Características", "🔀 Divisão e fuga de dados", "🧠 Modelo",
    "📊 Métricas e erros clínicos", "🎲 Incerteza e limites", "⬇️ Exportar",
])

# ----------------------------- RESUMO --------------------------------
with t_resumo:
    st.subheader("Visão geral da simulação")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Doentes sintéticos", df["pessoa"].nunique())
    m2.metric("Janelas totais", len(df))
    m3.metric("Treino", len(idx_tr))
    m4.metric("Validação", len(idx_va))
    m5.metric("Teste", len(idx_te))

    st.markdown("#### Onde cada ponto do enunciado é tratado")
    st.markdown(
        """
| Requisito do enunciado | Aba |
|---|---|
| Amostragem | 📡 Amostragem |
| Ruído e artefactos | 🔊 Ruído e artefactos |
| Filtragem | 🎛️ Filtragem |
| Segmentação e normalização | ✂️ Segmentação e normalização |
| Representação temporal e tempo-frequência | 〰️ Tempo-frequência |
| Características e aprendizagem automática | 🧮 Características · 🧠 Modelo |
| Separação por doente, fuga de dados, treino/validação/teste | 🔀 Divisão e fuga de dados |
| Métricas, desequilíbrio de classes | 👥 Coorte e classes · 📊 Métricas |
| Erros clinicamente relevantes | 📊 Métricas e erros clínicos |
| Limites e incerteza | 🎲 Incerteza e limites |
| Interpretação sem alegar uso clínico | Avisos em todas as abas |
        """
    )
    st.markdown("#### Métricas no conjunto de teste")
    cols = st.columns(len(met_te))
    for c, (k, v) in zip(cols, met_te.items()):
        c.metric(k, fmt(v))
    st.caption(f"Limiar de decisão actual: {limiar:.2f}. "
               "Cada parâmetro é alterado na aba correspondente.")

# ----------------------------- COORTE --------------------------------
with t_coorte:
    st.subheader("Coorte sintética e desequilíbrio de classes")
    a, b = st.columns(2)
    a.slider("Número de doentes sintéticos", 40, 200, DEFAULTS["n_pessoas"],
             step=10, key="n_pessoas")
    b.slider("Prevalência do Padrão B", 0.15, 0.50, DEFAULTS["prevalencia"],
             step=0.05, key="prevalencia",
             help="Fracção de doentes com Padrão B. Valores baixos criam desequilíbrio.")
    a.slider("Variação do ganho entre doentes", 0.0, 0.6, DEFAULTS["variab_ganho"],
             step=0.05, key="variab_ganho",
             help="Simula diferenças de amplitude entre doentes/eléctrodos.")
    b.number_input("Semente aleatória", 1, 50_000_000, DEFAULTS["semente"],
                   step=1, key="semente")
    st.slider("Doente de exemplo usado nas visualizações (1–8)", 1, 8,
              DEFAULTS["pessoa_exemplo"], key="pessoa_exemplo")
    st.caption(f"Exemplo seleccionado: **{pid_ex}** — {NOMES_CLASSES[ex['classe']]}.")

    dist = pd.DataFrame({
        "Doentes": df.groupby("classe")["pessoa"].nunique(),
        "Janelas": df.groupby("classe").size(),
    }).reindex([0, 1]).fillna(0)
    dist.index = NOMES_CLASSES
    st.bar_chart(dist)

    prev = float(np.mean(y_all == 1))
    if prev < 0.35:
        st.info(f"Desequilíbrio: apenas {prev:.0%} das janelas são Padrão B. "
                "A acurácia torna-se enganadora; veja a baseline e a acurácia "
                "balanceada na aba Métricas.")

# ----------------------------- AMOSTRAGEM ----------------------------
with t_amostr:
    st.subheader("Amostragem")
    a, b = st.columns(2)
    a.select_slider("Frequência de amostragem (Hz)", [125, 250, 360, 500],
                    DEFAULTS["fs"], key="fs")
    b.slider("Duração do registo por doente (s)", 20, 60, DEFAULTS["duracao"],
             step=5, key="duracao")
    m1, m2, m3 = st.columns(3)
    m1.metric("Amostras por registo", f"{len(ex['t'])}")
    m2.metric("Intervalo entre amostras", f"{1000 / fs:.1f} ms")
    m3.metric("Frequência de Nyquist", f"{fs / 2:.0f} Hz")

    fig, ax = plt.subplots(figsize=(12, 3.5))
    n = int(1.5 * fs)
    ax.plot(ex["t"][:n], ex["bruto"][:n], lw=0.8)
    if fs <= 250:
        ax.plot(ex["t"][:n], ex["bruto"][:n], ".", ms=3)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Amplitude (u.a.)")
    ax.set_title("Primeiros 1,5 s do sinal bruto (cada ponto é uma amostra)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Uma frequência baixa reduz o detalhe do complexo QRS (muito estreito); "
               "uma frequência alta aumenta o custo computacional. O filtro só pode "
               "actuar até à frequência de Nyquist.")

# ----------------------------- RUÍDO ---------------------------------
with t_ruido:
    st.subheader("Ruído e artefactos")
    a, b = st.columns(2)
    a.slider("Ruído aleatório (desvio padrão)", 0.0, 0.20, DEFAULTS["ruido"],
             step=0.005, key="ruido", help="No Padrão B o ruído é 1,5× maior.")
    b.slider("Deriva da linha de base (amplitude)", 0.0, 0.40, DEFAULTS["deriva"],
             step=0.01, key="deriva", help="Simula a respiração (0,25 Hz).")
    a.slider("Interferência da rede 50 Hz (amplitude)", 0.0, 0.20, DEFAULTS["rede"],
             step=0.005, key="rede")
    b.slider("Artefactos de movimento por registo", 0, 6, DEFAULTS["n_art"], key="n_art")
    a.slider("Amplitude dos artefactos", 0.1, 1.5, DEFAULTS["amp_art"],
             step=0.1, key="amp_art")

    resid = ex["bruto"] - ex["filtrado"]
    snr = 10 * np.log10(np.var(ex["filtrado"]) / (np.var(resid) + 1e-12))
    b.metric("Relação sinal/ruído aproximada", f"{snr:.1f} dB")

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(ex["t"], ex["bruto"], lw=0.7)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Amplitude (u.a.)")
    ax.set_title(f"Sinal bruto de {pid_ex} (ruído, deriva, 50 Hz e artefactos)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ----------------------------- FILTRAGEM -----------------------------
with t_filtro:
    st.subheader("Filtragem")
    a, b = st.columns(2)
    a.slider("Corte inferior (Hz)", 0.1, 2.0, DEFAULTS["low"], step=0.1, key="low")
    b.slider("Corte superior (Hz)", 20, 100, DEFAULTS["high"], step=5, key="high")
    a.slider("Ordem do filtro Butterworth", 1, 8, DEFAULTS["ordem"], key="ordem")
    b.checkbox("Filtro notch a 50 Hz", DEFAULTS["notch"], key="notch")
    if float(P("high")) > 0.95 * fs / 2:
        st.warning("O corte superior excede a frequência de Nyquist; foi limitado "
                   f"a {0.95 * fs / 2:.0f} Hz.")

    esq, dir_ = st.columns(2)
    with esq:
        sos = butter(int(P("ordem")), [float(P("low")), min(float(P("high")), 0.95 * fs / 2)],
                     btype="bandpass", fs=fs, output="sos")
        w, h = sosfreqz(sos, worN=2048, fs=fs)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(w, 20 * np.log10(np.abs(h) + 1e-12))
        ax.set_ylim(-60, 5)
        ax.set_xlim(0, min(100, fs / 2))
        ax.set_xlabel("Frequência (Hz)")
        ax.set_ylabel("Ganho (dB)")
        ax.set_title("Resposta em frequência do passa-banda")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with dir_:
        n = int(4 * fs)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(ex["t"][:n], ex["bruto"][:n], lw=0.7, alpha=0.7, label="Bruto")
        ax.plot(ex["t"][:n], ex["filtrado"][:n], lw=1.0, label="Filtrado")
        ax.set_xlabel("Tempo (s)")
        ax.set_title("Antes e depois (primeiros 4 s)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    st.caption("O filtro é aplicado ao registo completo antes de segmentar, "
               "o que evita efeitos de borda em cada janela.")

# ----------------------------- SEGMENTAÇÃO ---------------------------
with t_seg:
    st.subheader("Segmentação e normalização")
    a, b, c = st.columns(3)
    a.select_slider("Duração da janela (s)", [2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0],
                    DEFAULTS["janela_s"], key="janela_s")
    b.select_slider("Sobreposição entre janelas", [0.0, 0.25, 0.5],
                    DEFAULTS["sobreposicao"], key="sobreposicao",
                    format_func=lambda v: f"{int(v * 100)}%")
    c.selectbox("Normalização", NORMALIZACOES,
                NORMALIZACOES.index(DEFAULTS["normalizacao"]), key="normalizacao")

    janelas_ex = segmentar(ex["filtrado"], fs, float(P("janela_s")), float(P("sobreposicao")))
    st.caption(f"Cada doente gera {len(janelas_ex)} janelas. "
               "Janelas sobrepostas do mesmo doente são muito parecidas, "
               "por isso a divisão treino/teste tem de ser por doente.")

    esq, dir_ = st.columns(2)
    with esq:
        n_j = int(round(float(P("janela_s")) * fs))
        passo = max(1, int(round(n_j * (1 - float(P("sobreposicao"))))))
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(ex["t"], ex["filtrado"], lw=0.7)
        for k, i in enumerate(range(0, len(ex["t"]) - n_j + 1, passo)):
            ax.axvspan(ex["t"][i], ex["t"][i] + n_j / fs,
                       alpha=0.12, color="C1" if k % 2 else "C2")
        ax.set_xlabel("Tempo (s)")
        ax.set_title("Janelas sobre o registo filtrado")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with dir_:
        reg = normalizar_registo(ex["filtrado"], P("normalizacao"))
        jan = normalizar_janela(reg[:n_j], P("normalizacao"))
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(np.arange(n_j) / fs, ex["filtrado"][:n_j], lw=0.8, label="Filtrado")
        ax.plot(np.arange(n_j) / fs, jan, lw=1.0, label=f"Normalizado ({P('normalizacao')})")
        ax.set_xlabel("Tempo na janela (s)")
        ax.set_title("Primeira janela antes e depois da normalização")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    st.caption("A normalização por janela torna a média e o desvio padrão constantes, "
               "pelo que essas características deixam de ser informativas.")

# ----------------------------- TEMPO-FREQUÊNCIA ----------------------
with t_tf:
    st.subheader("Representação tempo-frequência")
    st.select_slider("Janela do espectrograma (amostras)", [64, 128, 256, 512],
                     DEFAULTS["nperseg"], key="nperseg")
    nps = int(P("nperseg"))
    esq, dir_ = st.columns(2)
    with esq:
        fig, ax = plt.subplots(figsize=(6, 4))
        for pid, e in exemplos.items():
            f_, p_ = welch(e["filtrado"], fs=fs, nperseg=min(len(e["filtrado"]), 512))
            ax.semilogy(f_, p_, lw=0.8, alpha=0.8,
                        color="C0" if e["classe"] == 0 else "C3")
        ax.plot([], [], color="C0", label="Padrão A")
        ax.plot([], [], color="C3", label="Padrão B")
        ax.set_xlim(0, 50)
        ax.set_xlabel("Frequência (Hz)")
        ax.set_ylabel("Densidade espectral")
        ax.set_title("Espectro (Welch) dos doentes de exemplo")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with dir_:
        f_, t_, S = spectrogram(ex["filtrado"], fs=fs, nperseg=nps, noverlap=nps * 3 // 4)
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.pcolormesh(t_, f_, 10 * np.log10(S + 1e-12), shading="auto")
        ax.set_ylim(0, 50)
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Frequência (Hz)")
        ax.set_title(f"Espectrograma de {pid_ex}")
        fig.colorbar(im, ax=ax, label="dB")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    st.caption("Janelas maiores dão melhor resolução em frequência e pior resolução no tempo; "
               "janelas menores fazem o inverso.")

# ----------------------------- CARACTERÍSTICAS -----------------------
with t_feat:
    st.subheader("Extracção de características")
    st.multiselect("Grupos de características", GRUPOS_FEAT,
                   DEFAULTS["grupos_feat"], key="grupos_feat")
    if not P("grupos_feat"):
        st.warning("Nenhum grupo seleccionado: foi usado o grupo 'Estatísticas temporais'.")
    st.caption(f"{len(colunas)} características por janela: " + ", ".join(colunas))

    n_lin = int(np.ceil(len(colunas) / 3))
    fig, axes = plt.subplots(n_lin, 3, figsize=(12, 2.6 * n_lin), squeeze=False)
    for ax, col in zip(axes.ravel(), colunas):
        ax.boxplot([df.loc[df.classe == 0, col], df.loc[df.classe == 1, col]])
        ax.set_xticks([1, 2])
        ax.set_xticklabels(NOMES_CLASSES, fontsize=8)
        ax.set_title(col, fontsize=9)
    for ax in axes.ravel()[len(colunas):]:
        ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Média por classe**")
    resumo = df.groupby("classe")[colunas].mean().T
    resumo.columns = NOMES_CLASSES
    st.dataframe(resumo)

# ----------------------------- DIVISÃO -------------------------------
with t_div:
    st.subheader("Divisão por doente e prevenção de fuga de dados")
    a, b = st.columns(2)
    a.slider("Proporção de doentes para teste", 0.15, 0.35, DEFAULTS["prop_teste"],
             step=0.05, key="prop_teste")
    b.slider("Proporção de doentes para validação", 0.10, 0.30, DEFAULTS["prop_val"],
             step=0.05, key="prop_val")

    pes_all = df["pessoa"].to_numpy()
    linhas = []
    for nome, idx in [("Treino", idx_tr), ("Validação", idx_va), ("Teste", idx_te)]:
        sub = df.iloc[idx]
        linhas.append({
            "Conjunto": nome,
            "Doentes": sub["pessoa"].nunique(),
            "Doentes Padrão B": sub.loc[sub.classe == 1, "pessoa"].nunique(),
            "Janelas": len(sub),
            "Janelas Padrão B": int((sub.classe == 1).sum()),
        })
    st.dataframe(pd.DataFrame(linhas), hide_index=True)

    conj = [set(pes_all[i]) for i in (idx_tr, idx_va, idx_te)]
    sem_fuga = not (conj[0] & conj[1] or conj[0] & conj[2] or conj[1] & conj[2])
    if sem_fuga:
        st.success("Verificação: nenhum doente aparece em mais do que um conjunto.")
    else:
        st.error("Fuga de dados: há doentes repetidos entre conjuntos.")
    st.caption("O treino ajusta o modelo, a validação serve para escolher o limiar "
               "de decisão e o teste só é usado para a avaliação final.")

    st.checkbox("Comparar com uma divisão ERRADA (por janela)", key="mostrar_fuga")
    if P("mostrar_fuga"):
        acc_e, auc_e = comparar_divisao_errada(
            df[colunas].to_numpy(), y_all, int(P("n_estimators")), int(P("max_depth")),
            P("class_weight"), float(P("prop_teste")), semente)
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Acurácia (por janela, errada)", fmt(acc_e))
        x2.metric("AUC (por janela, errada)", fmt(auc_e))
        x3.metric("Acurácia (por doente, correcta)", fmt(met_te["Acurácia"]))
        x4.metric("AUC (por doente, correcta)", fmt(met_te["AUC-ROC"]))
        st.caption("A divisão por janela costuma dar resultados mais optimistas, porque "
                   "janelas do mesmo doente aparecem no treino e no teste.")

# ----------------------------- MODELO --------------------------------
with t_modelo:
    st.subheader("Modelo de aprendizagem automática (Random Forest)")
    a, b, c = st.columns(3)
    a.slider("Número de árvores", 50, 500, DEFAULTS["n_estimators"], step=50, key="n_estimators")
    b.slider("Profundidade máxima (0 = sem limite)", 0, 20, DEFAULTS["max_depth"], key="max_depth")
    c.selectbox("Pesos das classes", ["balanced", "nenhum"],
                ["balanced", "nenhum"].index(DEFAULTS["class_weight"]), key="class_weight",
                help="'balanced' compensa o desequilíbrio de classes.")

    m1, m2, m3 = st.columns(3)
    auc_tr = roc_auc_score(y_tr, R["p_tr"]) if len(np.unique(y_tr)) == 2 else float("nan")
    m1.metric("AUC no treino", fmt(auc_tr))
    m2.metric("AUC na validação", fmt(met_va["AUC-ROC"]))
    m3.metric("AUC no teste", fmt(met_te["AUC-ROC"]))
    st.caption("Uma grande diferença entre treino e validação indica sobreajuste.")

    imp = pd.Series(R["importancia"], index=colunas).sort_values()
    st.markdown("**Importância das características**")
    st.bar_chart(imp)

# ----------------------------- MÉTRICAS ------------------------------
with t_metr:
    st.subheader("Métricas e erros clinicamente relevantes")
    a, b = st.columns(2)
    a.checkbox("Escolher o limiar na validação (sensibilidade alvo)",
               DEFAULTS["usar_limiar_val"], key="usar_limiar_val")
    a.slider("Sensibilidade alvo", 0.50, 0.99, DEFAULTS["sens_alvo"], step=0.01,
             key="sens_alvo", disabled=not P("usar_limiar_val"))
    b.slider("Limiar de decisão (probabilidade de Padrão B)", 0.05, 0.95,
             DEFAULTS["limiar"], step=0.05, key="limiar",
             disabled=bool(P("usar_limiar_val")))
    b.slider("Custo de um falso negativo (relativo a um falso positivo = 1)",
             1, 20, DEFAULTS["custo_fn"], key="custo_fn")
    st.caption(f"Limiar em uso: **{limiar:.2f}**")

    tab_m = pd.DataFrame({"Validação": met_va, "Teste": met_te})
    cols = st.columns(4)
    for i, (k, v) in enumerate(met_te.items()):
        cols[i % 4].metric(k, fmt(v))
    with st.expander("Comparar validação e teste"):
        st.dataframe(tab_m.style.format("{:.3f}"))

    esq, dir_ = st.columns([1, 2])
    with esq:
        cm = np.array([[tn, fp], [fn, tp]])
        fig, ax = plt.subplots(figsize=(4, 3.6))
        ConfusionMatrixDisplay(cm, display_labels=NOMES_CLASSES).plot(
            ax=ax, values_format="d", colorbar=False)
        ax.set_title("Matriz de confusão (teste)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with dir_:
        if len(np.unique(y_te)) == 2:
            fpr, tpr, _ = roc_curve(y_te, R["p_te"])
            prc, rec, _ = precision_recall_curve(y_te, R["p_te"])
            ths = np.linspace(0.02, 0.98, 49)
            ss = [calcular_metricas(y_te, R["p_te"], t)[0] for t in ths]
            fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
            axs[0].plot(fpr, tpr)
            axs[0].plot([0, 1], [0, 1], "--", color="gray")
            axs[0].set_title("Curva ROC")
            axs[0].set_xlabel("1 − Especificidade")
            axs[0].set_ylabel("Sensibilidade")
            axs[1].plot(rec, prc)
            axs[1].set_title("Precisão–Recall")
            axs[1].set_xlabel("Sensibilidade")
            axs[1].set_ylabel("Precisão")
            axs[2].plot(ths, [s["Sensibilidade"] for s in ss], label="Sensibilidade")
            axs[2].plot(ths, [s["Especificidade"] for s in ss], label="Especificidade")
            axs[2].axvline(limiar, color="gray", ls="--")
            axs[2].set_title("Efeito do limiar")
            axs[2].set_xlabel("Limiar")
            axs[2].legend(fontsize=8)
            for ax in axs:
                ax.grid(alpha=0.25)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("O conjunto de teste tem só uma classe; curvas indisponíveis.")

    st.markdown("#### Erros e custo")
    custo = int(P("custo_fn")) * fn + fp
    custo_tudo_a = int(P("custo_fn")) * int((y_te == 1).sum())
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Falsos negativos (B perdido)", fn)
    e2.metric("Falsos positivos (falso alarme)", fp)
    e3.metric("Custo do modelo", custo)
    e4.metric("Custo se previsse sempre A", custo_tudo_a)

    base = max(np.mean(y_te == 0), np.mean(y_te == 1))
    sens, esp = met_te["Sensibilidade"], met_te["Especificidade"]
    st.markdown("#### Interpretação")
    st.markdown(
        f"- Prever sempre a classe maioritária daria acurácia de **{base:.3f}**; "
        f"o modelo obtém **{met_te['Acurácia']:.3f}**. "
        "Em classes desequilibradas, compare também a acurácia balanceada.\n"
        f"- Sensibilidade **{fmt(sens, 2)}**: de 100 janelas de Padrão B, o modelo "
        f"deteta cerca de {fmt(sens * 100 if not np.isnan(sens) else sens, 0)} "
        "e falha as restantes (falsos negativos).\n"
        f"- Especificidade **{fmt(esp, 2)}**: de 100 janelas de Padrão A, cerca de "
        f"{fmt((1 - esp) * 100 if not np.isnan(esp) else esp, 0)} geram falso alarme.\n"
        "- Baixar o limiar aumenta a sensibilidade mas aumenta os falsos alarmes.\n"
        "- Estes valores descrevem apenas dados sintéticos; **não têm significado clínico**."
    )

# ----------------------------- INCERTEZA -----------------------------
with t_inc:
    st.subheader("Incerteza e limites")
    st.slider("Reamostragens bootstrap", 100, 1000, DEFAULTS["n_boot"], step=100, key="n_boot")

    ic = ic_bootstrap(y_te, R["p_te"], pes_te, limiar, int(P("n_boot")), semente)
    st.markdown("**Intervalos de confiança a 95 % (bootstrap por doente, teste)**")
    linhas = [{"Métrica": k, "Valor": met_te[k], "IC 95% inferior": lo, "IC 95% superior": hi}
              for k, (lo, hi) in ic.items()]
    st.dataframe(pd.DataFrame(linhas).style.format(
        {"Valor": "{:.3f}", "IC 95% inferior": "{:.3f}", "IC 95% superior": "{:.3f}"}),
        hide_index=True)
    n_b_te = df.iloc[idx_te].loc[lambda d: d.classe == 1, "pessoa"].nunique()
    st.caption(f"O teste tem {len(np.unique(pes_te))} doentes, dos quais {n_b_te} com Padrão B. "
               "Poucos doentes produzem intervalos largos.")

    esq, dir_ = st.columns(2)
    with esq:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(R["s_te"], bins=20)
        ax.set_xlabel("Desvio padrão das previsões das árvores")
        ax.set_ylabel("Nº de janelas")
        ax.set_title("Incerteza do modelo por janela")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with dir_:
        zona = float(np.mean((R["p_te"] > 0.35) & (R["p_te"] < 0.65)))
        st.metric("Janelas na zona incerta (0,35 a 0,65)", f"{zona:.0%}")
        st.caption("Janelas nesta zona seriam candidatas a revisão humana.")

    st.markdown("#### Limites deste estudo")
    st.markdown(
        "- Os sinais e os rótulos são criados pelo mesmo gerador: o modelo aprende "
        "as regras do simulador, não doença real.\n"
        "- Há poucos doentes; resultados mudam com a semente.\n"
        "- As características são simples e não substituem a análise morfológica do ECG.\n"
        "- Não há validação externa nem dados de outro equipamento.\n"
        "- Nenhum resultado desta aplicação deve apoiar decisões clínicas."
    )

# ----------------------------- EXPORTAR ------------------------------
with t_exp:
    st.subheader("Descarregar resultados")
    sub = pd.concat([
        pd.DataFrame({"Conjunto": "Validação", "Pessoa": pes_all[idx_va],
                      "Janela": df["janela"].to_numpy()[idx_va], "Classe real": y_va,
                      "Probabilidade de Padrão B": R["p_va"], "Incerteza (desvio árvores)": R["s_va"]}),
        pd.DataFrame({"Conjunto": "Teste", "Pessoa": pes_te,
                      "Janela": df["janela"].to_numpy()[idx_te], "Classe real": y_te,
                      "Probabilidade de Padrão B": R["p_te"], "Incerteza (desvio árvores)": R["s_te"]}),
    ], ignore_index=True)
    sub["Classe prevista"] = (sub["Probabilidade de Padrão B"] >= limiar).astype(int)

    parametros = {k: (list(P(k)) if isinstance(P(k), (list, tuple)) else P(k)) for k in DEFAULTS}
    exportacao = {
        "parametros": parametros,
        "limiar_usado": limiar,
        "metricas_teste": {k: (None if np.isnan(v) else float(v)) for k, v in met_te.items()},
        "metricas_validacao": {k: (None if np.isnan(v) else float(v)) for k, v in met_va.items()},
        "matriz_confusao_teste_[[TN,FP],[FN,TP]]": [[tn, fp], [fn, tp]],
        "aviso": "Dados e rótulos sintéticos; métricas sem significado clínico.",
    }
    relatorio = classification_report(
        y_te, (R["p_te"] >= limiar).astype(int), labels=[0, 1],
        target_names=NOMES_CLASSES, zero_division=0)

    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("⬇️ Dados (CSV)", df.to_csv(index=False).encode("utf-8"),
                       "dados_ecg_sinteticos.csv", "text/csv")
    c2.download_button("⬇️ Previsões (CSV)", sub.to_csv(index=False).encode("utf-8"),
                       "previsoes.csv", "text/csv")
    c3.download_button("⬇️ Métricas e parâmetros (JSON)",
                       json.dumps(exportacao, ensure_ascii=False, indent=2),
                       "metricas_parametros.json", "application/json")
    c4.download_button("⬇️ Relatório (TXT)", relatorio,
                       "relatorio_classificacao.txt", "text/plain")
    st.code(relatorio, language="text")

st.divider()
st.caption("T11 | Demonstração académica de análise de séries temporais biomédicas.")
