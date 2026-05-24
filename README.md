# 🎬 SentimentLab — Análise Comparativa de ML para Sentimentos

> ANÁLISE COMPARATIVA DE ALGORITMOS CLÁSSICOS DE MACHINE LEARNING APLICADOS À CLASSIFICAÇÃO DE SENTIMENTOS EM TEXTOS
> Trabalho de Conclusão de Curso — Engenharia de Software · UNINTER · 2026  
> **Janyelle Oliveira** &
> **Júlia Oliveira**

---

## 📌 Sobre o Projeto

O **SentimentLab** é uma aplicação web interativa desenvolvida como parte do TCC de Engenharia de Software da UNINTER. O sistema permite analisar o sentimento de avaliações de filmes utilizando diferentes algoritmos de Machine Learning, comparando seus resultados de forma visual e interativa.

O estudo realiza uma **análise comparativa** entre 4 algoritmos clássicos de ML combinados com 2 estratégias de vetorização de texto, totalizando **8 experimentos** controlados sobre o dataset IMDB 50K Movie Reviews.

---

## 🎯 Questão de Pesquisa

> *Qual algoritmo clássico de Machine Learning apresenta melhor desempenho na classificação de sentimentos em avaliações de filmes, e como a escolha da estratégia de vetorização impacta esse resultado?*

---

## 🤖 Modelos Comparados

| Algoritmo | Vetorizações | Experimentos |
|---|---|---|
| Naive Bayes Multinomial | TF-IDF, BoW | 2 |
| Regressão Logística | TF-IDF, BoW | 2 |
| SVM (LinearSVC) | TF-IDF, BoW | 2 |
| Random Forest | TF-IDF, BoW | 2 |
| **Total** | | **8 experimentos** |

---

## 📊 Resultados

| Modelo | Vetorização | Acurácia | F1-Score |
|---|---|---|---|
| **Regressão Logística** | **TF-IDF** | **0.8883** | **0.8893** |
| SVM | TF-IDF | 0.8826 | 0.8834 |
| Regressão Logística | BoW | 0.8709 | 0.8713 |
| Naive Bayes | TF-IDF | 0.8513 | 0.8518 |
| Random Forest | BoW | 0.8475 | 0.8470 |
| Random Forest | TF-IDF | 0.8487 | 0.8467 |
| SVM | BoW | 0.8437 | 0.8434 |
| Naive Bayes | BoW | 0.8426 | 0.8406 |

**🏆 Melhor modelo:** Regressão Logística + TF-IDF (F1-Score: 0.8893)

---

## 🗂️ Estrutura do Projeto

```
tcc-sentiment-analysis/
├── notebooks/
│   ├── 01_EDA.ipynb               # Análise exploratória do dataset
│   ├── 02_preprocessing.ipynb     # Pré-processamento dos textos
│   ├── 03_models.ipynb            # Treinamento dos 8 experimentos
│   └── 04_results.ipynb           # Visualização e análise dos resultados
├── results/
│   └── resultados.csv             # Métricas dos 8 experimentos
├── app.py                         # Interface Streamlit
├── requirements.txt               # Dependências do projeto
├── .gitignore
└── README.md
```

> ⚠️ As pastas `data/` e `models/` não estão no repositório por conterem arquivos grandes. Siga as instruções abaixo para reproduzir o projeto.

---

## ⚙️ Como Reproduzir

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/tcc-sentiment-analysis.git
cd tcc-sentiment-analysis
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Baixe o dataset

Acesse o link abaixo e faça o download do arquivo `IMDB Dataset.csv`:  
🔗 [IMDB Dataset of 50K Movie Reviews — Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

Coloque o arquivo na pasta:
```
data/IMDB Dataset.csv
```

### 4. Execute os notebooks em ordem

```
01_EDA.ipynb → 02_preprocessing.ipynb → 03_models.ipynb → 04_results.ipynb
```

> Os modelos treinados serão salvos automaticamente na pasta `models/` ao executar o notebook 03.

### 5. Rode a aplicação

```bash
streamlit run app.py
```

---

## 🛠️ Tecnologias Utilizadas

| Tecnologia | Finalidade |
|---|---|
| Python 3 | Linguagem principal |
| Streamlit | Interface web interativa |
| scikit-learn | Algoritmos de ML e métricas |
| pandas | Manipulação de dados |
| NLTK | Pré-processamento de texto |
| matplotlib / seaborn | Visualização de dados |
| joblib | Serialização dos modelos |
| Google Colab | Ambiente de desenvolvimento |
| GitHub | Versionamento de código |

---

## 🧹 Pipeline de Pré-processamento

1. Remoção de tags HTML (`<br />`)
2. Tratamento de apóstrofos
3. Remoção de caracteres especiais e números
4. Conversão para minúsculas
5. Tokenização
6. Remoção de stopwords (NLTK — inglês)
7. Stemming (PorterStemmer)

---

## 📱 Funcionalidades da Aplicação

- **Analisar Texto:** cole qualquer avaliação e receba a classificação (Positivo/Negativo) com percentual de confiança
- **Comparar todos os modelos:** veja a predição de todos os 4 algoritmos lado a lado
- **Relatório de Desempenho:** tabela comparativa com todas as métricas, gráfico de F1-Score e matriz de confusão do melhor modelo

---

## 👩‍💻 Autoras

| | |
|---|---|
| **Janyelle Oliveira** | Análise exploratória, pré-processamento, métricas e resultados |
| **Júlia Oliveira** | Vetorização, treinamento dos modelos e interface Streamlit |

---

## 📄 Licença

Este projeto foi desenvolvido para fins acadêmicos como Trabalho de Conclusão de Curso.  
UNINTER — Centro Universitário Internacional · 2026
