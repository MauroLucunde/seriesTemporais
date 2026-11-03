# G11 — Aplicação Streamlit para ECG sintético

## Requisitos
- Python 3.10 ou superior
- Ligação à internet para instalar as bibliotecas (não é necessária para executar depois de instaladas)

## Instalação
Abra o terminal na pasta do projecto e execute:

```bash
python -m venv .venv
```

Activar o ambiente virtual:

**Windows**
```bash
.venv\Scripts\activate
```

**Linux/macOS**
```bash
source .venv/bin/activate
```

Instalar dependências:

```bash
python -m pip install -r requirements.txt
```

## Executar a aplicação

```bash
streamlit run app.py
```

O Streamlit abrirá a aplicação no navegador. Se não abrir automaticamente, aceda ao endereço indicado no terminal (normalmente http://localhost:8501).

## Funcionalidades
- Geração configurável de sinais ECG sintéticos;
- Filtragem Butterworth passa-banda;
- Extracção de características;
- Treino e avaliação de Random Forest;
- Separação treino/teste por pessoa para reduzir fuga de dados;
- Gráficos, matriz de confusão e relatório de classificação;
- Exportação de dados, previsões e métricas.

## Aviso
Este é um projecto didáctico. Os sinais e rótulos são artificiais. Não utilizar para diagnóstico, triagem ou decisões clínicas.
