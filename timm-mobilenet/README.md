# Timm MobileNetV3

This project uses a pre-trained MobileNetV3 model from the `timm` library to classify an image and displays the top 5 predictions.

## Estrutura do Projeto

O projeto foi reestruturado para um ambiente mais profissional, com a seguinte organização:

```
timm-mobilenet/
├── images/
│   └── bird.jpg
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── plotting.py
│   └── utils.py
├── main.py
├── pyproject.toml
└── README.md
```

- **`main.py`**: Ponto de entrada da aplicação.
- **`src/config.py`**: Carrega o modelo pré-treinado e suas configurações.
- **`src/model.py`**: Realiza a predição na imagem.
- **`src/plotting.py`**: Plota a imagem de entrada e o gráfico com as 5 melhores predições.
- **`src/utils.py`**: Funções utilitárias, como carregar a imagem e obter os rótulos do ImageNet.

## Como executar

1.  **Crie um ambiente virtual e ative-o:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # No Windows, use `.venv\Scripts\activate`
    ```

2.  **Instale as dependências:**

    ```bash
    pip install -r requirements.txt
    ```
    > **Obs:** O `requirements.txt` será criado no próximo passo.

3.  **Execute o script principal:**

    ```bash
    python main.py
    ```

Isso abrirá uma janela mostrando a imagem de entrada e um gráfico de barras com as 5 principais classificações e suas probabilidades.
