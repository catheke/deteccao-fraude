import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image
import os

# Configuração da página
st.set_page_config(page_title="Detecção de Fraudes - Pedro Calenga", layout="wide", page_icon=":credit_card:")

# Estilo CSS personalizado com tema escuro
st.markdown("""
<style>
    .main {background: linear-gradient(to bottom, #1e3a8a, #0f172a);}
    .stTabs [data-baseweb="tab-list"] {background-color: #1e3a8a; padding: 10px; border-radius: 8px;}
    .stTabs [data-baseweb="tab"] {color: #ffffff; font-weight: bold; padding: 10px 20px; border-radius: 8px;}
    .stTabs [data-baseweb="tab"]:hover {background-color: #06b6d4; color: #ffffff;}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {background-color: #06b6d4; color: #ffffff;}
    .stButton>button {background-color: #06b6d4; color: white; border-radius: 8px; padding: 10px 20px; font-weight: bold; transition: all 0.3s;}
    .stButton>button:hover {background-color: #3b82f6; transform: scale(1.05);}
    .stNumberInput input {border-radius: 8px; border: 1px solid #06b6d4; background-color: #172554; color: #ffffff; padding: 8px;}
    .stSuccess {background-color: #064e3b; border: 1px solid #34c759; padding: 15px; border-radius: 8px; color: #ffffff; font-weight: bold;}
    .stError {background-color: #7f1d1d; border: 1px solid #ef4444; padding: 15px; border-radius: 8px; color: #ffffff;}
    h1 {color: #ffffff; text-align: center; font-size: 2.5em; margin-bottom: 20px;}
    h2 {color: #06b6d4; font-size: 1.8em;}
    p {color: #d1d5db;}
    .content-box {background-color: #172554; padding: 20px; border-radius: 8px; border: 1px solid #06b6d4;}
    .footer {text-align: center; color: #d1d5db; margin-top: 40px; font-size: 0.9em;}
    .stSelectbox div[data-baseweb="select"] {border-radius: 8px; border: 1px solid #06b6d4; background-color: #172554; color: #ffffff;}
    .stSelectbox div[data-baseweb="select"] span {color: #ffffff;}
</style>
""", unsafe_allow_html=True)

# Título principal
st.title("Detecção de Fraudes em Cartões de Crédito")
st.markdown("<h3 style='text-align: center; color: #ffffff;'>Desenvolvido por Pedro Calenga</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #d1d5db;'>3º Ano de Ciência da Computação, Universidade Mandume Ya Ndemufayo, Instituto Politécnico da Huíla</p>", unsafe_allow_html=True)

# Carregar modelo e scaler
try:
    model = joblib.load('model/modelo_fraude.pkl')
    scaler = joblib.load('model/scaler.pkl')
except:
    st.error("Erro ao carregar 'modelo_fraude.pkl' ou 'scaler.pkl' da pasta 'model'. Verifique os arquivos.")
    st.stop()

# Dados pré-carregados para testes rápidos (4 transações: 2 normais, 2 fraudes)
test_data = {
    0: {'V1': -1.359807, 'V2': -0.072781, 'V3': 2.536347, 'V4': 1.378155, 'V5': -0.338321, 'V6': 0.462388, 'V7': 0.239599, 'V8': 0.098698, 'V9': 0.363787, 'V10': 0.090794, 'V11': -0.551600, 'V12': -0.617801, 'V13': -0.991390, 'V14': -0.311169, 'V15': 1.468177, 'V16': -0.470401, 'V17': 0.207971, 'V18': 0.025791, 'V19': 0.403993, 'V20': 0.251412, 'V21': -0.018307, 'V22': 0.277838, 'V23': -0.110474, 'V24': 0.066928, 'V25': 0.128539, 'V26': -0.189115, 'V27': 0.133558, 'V28': -0.021053, 'Amount': 149.62, 'Class': 0},
    1: {'V1': 1.191857, 'V2': 0.266151, 'V3': 0.166480, 'V4': 0.448154, 'V5': 0.060018, 'V6': -0.082361, 'V7': -0.078803, 'V8': 0.085102, 'V9': -0.255425, 'V10': -0.166974, 'V11': 1.612726, 'V12': 1.065235, 'V13': 0.489095, 'V14': -0.143772, 'V15': 0.635558, 'V16': 0.463917, 'V17': -0.114805, 'V18': -0.183361, 'V19': -0.145783, 'V20': -0.069083, 'V21': -0.225775, 'V22': -0.638672, 'V23': 0.101288, 'V24': -0.339846, 'V25': 0.167170, 'V26': 0.125895, 'V27': -0.008983, 'V28': 0.014724, 'Amount': 2.69, 'Class': 0},
    541: {'V1': -2.312227, 'V2': 1.951992, 'V3': -1.609851, 'V4': 3.997906, 'V5': -0.522188, 'V6': -1.426545, 'V7': -2.537387, 'V8': 1.391657, 'V9': -2.770089, 'V10': -2.772272, 'V11': 3.202033, 'V12': -2.899907, 'V13': -0.595222, 'V14': -4.289254, 'V15': 0.389724, 'V16': -1.140747, 'V17': -2.830056, 'V18': -0.016822, 'V19': 0.416956, 'V20': 0.126911, 'V21': 0.517232, 'V22': -0.035049, 'V23': -0.465211, 'V24': 0.320198, 'V25': 0.044519, 'V26': 0.177840, 'V27': 0.261145, 'V28': -0.143276, 'Amount': 0.00, 'Class': 1},
    623: {'V1': -3.043541, 'V2': -3.157307, 'V3': 1.088463, 'V4': 2.288644, 'V5': 1.359805, 'V6': -1.064823, 'V7': 0.325574, 'V8': -0.067794, 'V9': -0.270953, 'V10': -0.838587, 'V11': -0.414575, 'V12': -0.503141, 'V13': 0.676502, 'V14': -1.692029, 'V15': 2.000635, 'V16': 0.666780, 'V17': 0.599717, 'V18': 1.725321, 'V19': 0.283345, 'V20': 2.102339, 'V21': 0.661696, 'V22': 0.435477, 'V23': 1.375966, 'V24': -0.293803, 'V25': 0.279798, 'V26': -0.145362, 'V27': -0.252773, 'V28': 0.035764, 'Amount': 529.00, 'Class': 1}
}

# Criar abas
tab1, tab2, tab3, tab4 = st.tabs(["Introdução", "Previsão Manual", "Testes Rápidos", "Desempenho"])

# Aba 1: Introdução
with tab1:
    st.header("Introdução")
    st.markdown("""
    <div class='content-box'>
        <h2>Bem-vindo ao Sistema de Detecção de Fraudes</h2>
        <p>Desenvolvido por <strong>Pedro Calenga</strong>, estudante do 3º ano de Ciência da Computação, 
        <em>Universidade Mandume Ya Ndemufayo, Instituto Politécnico da Huíla</em>.</p>
        <p>Este projeto utiliza um modelo <strong>Random Forest</strong> treinado em transações de cartões de crédito 
        para detectar fraudes. Ele analisa 28 features anonimizadas (<code>V1</code> a <code>V28</code>) e o valor da transação 
        (<code>Amount</code>) para prever se uma transação é <strong>Normal</strong> (0) ou <strong>Fraudulenta</strong> (1).</p>
        <p><strong>Funcionalidades:</strong></p>
        <ul>
            <li><strong>Previsão Manual</strong>: Insira valores para testar novas transações.</li>
            <li><strong>Testes Rápidos</strong>: Use transações pré-carregadas para verificar o modelo.</li>
            <li><strong>Desempenho</strong>: Veja a matriz de confusão e a curva ROC para avaliar o modelo.</li>
        </ul>
        <p><strong>Desempenho do Modelo:</strong> Precision: 0,97 | Recall: 0,77 | F1-Score: 0,86 | AUC: ~0,95</p>
    </div>
    """, unsafe_allow_html=True)

# Aba 2: Previsão Manual
with tab2:
    st.header("Previsão Manual")
    st.markdown("""
    <div class='content-box'>
        <p><strong>Instruções:</strong> Insira valores para as 28 features (<code>V1</code> a <code>V28</code>), que são numéricos 
        entre -30 e 30 (obtidos por PCA), e o <code>Amount</code> (valor da transação, ex.: 100.50). Clique em "Prever" para verificar 
        se a transação é normal ou fraudulenta.</p>
        <p><strong>Dica:</strong> Use valores semelhantes aos do dataset, como -5 a 5 para <code>V1</code> a <code>V28</code>, e valores 
        positivos para <code>Amount</code>.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("form_previsao"):
        st.subheader("Insira os Dados da Transação")
        col1, col2 = st.columns(2)
        inputs = {}
        with col1:
            for i in range(1, 15):
                inputs[f'V{i}'] = st.number_input(f'V{i}', value=0.0, step=0.1, format="%.4f", key=f'v{i}')
        with col2:
            for i in range(15, 29):
                inputs[f'V{i}'] = st.number_input(f'V{i}', value=0.0, step=0.1, format="%.4f", key=f'v{i}')
            inputs['Amount'] = st.number_input('Amount (Valor da Transação)', value=0.0, min_value=0.0, step=0.01, key='amount')
        
        submitted = st.form_submit_button("Prever Transação", use_container_width=True)
        if submitted:
            data = np.array([[inputs[f'V{i}'] for i in range(1, 29)] + [inputs['Amount']]])
            data[:, -1] = scaler.transform(data[:, -1].reshape(-1, 1))  # Normalizar Amount
            prediction = model.predict(data)[0]
            proba = model.predict_proba(data)[0]
            st.success(f"Previsão: {'Fraude' if prediction == 1 else 'Normal'}")
            st.markdown(f"Probabilidade de Fraude: {proba[1]:.2%}")
            st.markdown(f"Probabilidade de Normal: {proba[0]:.2%}")

# Aba 3: Testes Rápidos
with tab3:
    st.header("Testes Rápidos")
    st.markdown("""
    <div class='content-box'>
        <p><strong>Instruções:</strong> Selecione uma transação pré-carregada para testar o modelo rapidamente. 
        As transações incluem exemplos de normais (Classe 0) e fraudes (Classe 1).</p>
    </div>
    """, unsafe_allow_html=True)
    
    option = st.selectbox("Escolha uma transação", 
                          options=[f"Transação {i} ({'Fraude' if test_data[i]['Class'] == 1 else 'Normal'})" for i in test_data.keys()], 
                          key='testes_rapidos')
    index = int(option.split()[1])
    
    if st.button("Prever Transação Selecionada", use_container_width=True):
        data = np.array([[test_data[index][f'V{i}'] for i in range(1, 29)] + [test_data[index]['Amount']]])
        data[:, -1] = scaler.transform(data[:, -1].reshape(-1, 1))  # Normalizar Amount
        prediction = model.predict(data)[0]
        proba = model.predict_proba(data)[0]
        st.success(f"Previsão: {'Fraude' if prediction == 1 else 'Normal'}")
        st.markdown(f"Probabilidade de Fraude: {proba[1]:.2%}")
        st.markdown(f"Probabilidade de Normal: {proba[0]:.2%}")
        st.markdown("Dados da Transação Selecionada:")
        st.dataframe(pd.DataFrame([test_data[index]]))

# Aba 4: Desempenho
with tab4:
    st.header("Desempenho")
    st.markdown("""
    <div class='content-box'>
        <p>O modelo Random Forest foi avaliado com:</p>
        <ul>
            <li><strong>Matriz de Confusão</strong>: Mostra acertos (transações normais e fraudes classificadas corretamente) 
            e erros (falsos positivos e falsos negativos).</li>
            <li><strong>Curva ROC</strong>: Avalia a capacidade do modelo de distinguir fraudes de normais. 
            A AUC próxima de 1 indica excelente desempenho.</li>
        </ul>
        <p><strong>Métricas no conjunto de teste:</strong></p>
        <ul>
            <li>Precision (Fraude): 0,97</li>
            <li>Recall (Fraude): 0,77</li>
            <li>F1-Score (Fraude): 0,86</li>
            <li>AUC: ~0,95</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Matriz de Confusão")
        try:
            matriz_img = Image.open('model/matriz_confusao.png')
            st.image(matriz_img, caption="Matriz de Confusão: [TN=56862, FP=2], [FN=23, TP=75]", use_column_width=True)
        except:
            st.error("Erro ao carregar 'matriz_confusao.png' da pasta 'model'. Verifique o arquivo.")
    with col2:
        st.subheader("Curva ROC")
        try:
            roc_img = Image.open('model/curva_roc.png')
            st.image(roc_img, caption="Curva ROC com AUC", use_column_width=True)
        except:
            st.error("Erro ao carregar 'curva_roc.png' da pasta 'model'. Verifique o arquivo.")

# Rodapé
st.markdown("""
<div class='footer'>
    <hr style='border-top: 2px solid #06b6d4;'>
    <p><strong>Desenvolvido por Pedro Calenga</strong></p>
    <p>3º Ano de Ciência da Computação, Universidade Mandume Ya Ndemufayo, Instituto Politécnico da Huíla</p>
    <p>Projeto: Detecção de Fraudes em Cartões de Crédito usando Random Forest</p>
    <p>Otimizado para Streamlit Cloud</p>
</div>
""", unsafe_allow_html=True)