# SpeechToTextSentiment

Este projeto desenvolve um sistema de classificação de sentimentos através de áudios utilizando Inteligência Artificial. O objetivo é analisar áudios de fala e identificar as emoções expressas, como felicidade, tristeza, raiva, surpresa, entre outros. Utilizando técnicas de aprendizado de máquina e redes neurais, o modelo é treinado em um conjunto diversificado de amostras de áudio para reconhecer e classificar diferentes sentimentos.

## Bibliotecas Utilizadas

- [numpy](https://numpy.org/): Biblioteca para suporte a arrays e matrizes multidimensionais.
- [pandas](https://pandas.pydata.org/): Biblioteca para manipulação e análise de dados.
- [librosa](https://librosa.org/): Biblioteca para análise de música e áudio em Python.
- [scikit-learn](https://scikit-learn.org/stable/): Biblioteca para aprendizado de máquina em Python.
- [tensorflow](https://www.tensorflow.org/): Plataforma de código aberto para machine learning.
- [keras](https://keras.io/): Biblioteca de redes neurais de alto nível para Python, rodando sobre TensorFlow.

## Funções Utilizadas

### `load_audio_files(directory)`
Carrega e pré-processa os arquivos de áudio a partir de um diretório, convertendo os áudios em espectrogramas e normalizando os valores.

### `extract_features(audio_data)`
Extrai características relevantes dos dados de áudio, como MFCCs (Mel-frequency cepstral coefficients), que são úteis para a análise de sentimentos na fala.

### `create_model(input_shape, num_classes)`
Cria uma arquitetura de rede neural convolucional (CNN) para a classificação de sentimentos, usando a API Keras. A arquitetura específica pode variar dependendo dos parâmetros fornecidos.

### `train_model(model, X_train, y_train, X_val, y_val)`
Treina o modelo de classificação de sentimentos usando os dados de treinamento e validação fornecidos, utilizando otimizadores e funções de perda apropriados.

### `evaluate_model(model, X_test, y_test)`
Avalia o desempenho do modelo utilizando os dados de teste fornecidos e retorna métricas de avaliação como precisão, recall e pontuação F1.

## Detalhes do Projeto

Este projeto visa criar um classificador eficaz de sentimentos em áudios utilizando técnicas de Aprendizado de Máquina e redes neurais convolucionais. Ele envolve a pré-processamento de dados de áudio, extração de características e treinamento de modelos para a classificação de sentimentos.

## Exemplo de Uso

```python
# Apenas funciona com python@3.9 para baixo
import speech_recognition as sr
from transformers import pipeline

# Carregar o áudio
def ouvirMic():
  # habilitar mic
  microfone = sr.Recognizer()
  print("Diga alguma coisa: ")
  with sr.Microphone() as source:
    # armazena o audio em texto
    audio = microfone.listen(source)
  try:
    frase = microfone.recognize_google(audio, language="pt-BR")
    return frase
  except sr.UnknownValueError:
    print("Não entendi")
  return False

# Analisar e Classificar
def classification():
  analise_sentimentos = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
  texto = ouvirMic()
  print(texto)
  propiedades = analise_sentimentos(texto)
  if propiedades[0]['label'] == "5 stars": 
    return print("\nALEGRE\n⭐⭐⭐⭐⭐")
  elif propiedades[0]['label'] == "4 stars": 
    return print("\nCONTENTE\n⭐⭐⭐⭐")
  elif propiedades[0]['label'] == "3 stars": 
    return print("\nNEUTRO\n⭐⭐⭐")
  elif propiedades[0]['label'] == "2 stars": 
    return print("\nTRISTE\n⭐⭐")
  else: 
    return print("\nDEPLORAVEL\n⭐")

# Resultado
classification()
