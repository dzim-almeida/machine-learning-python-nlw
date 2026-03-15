# Detecção de Objetos com YOLOS-small

Este projeto demonstra o uso do modelo YOLOS-small (You Only Look at One Sequence) para detecção de objetos em imagens. O modelo é carregado a partir do Hugging Face Hub e utilizado para identificar objetos em uma imagem de exemplo.

## Pré-requisitos

- Python 3.9 ou superior
- `pip` (gerenciador de pacotes do Python)

## Instalação

1. **Clone o repositório:**
   ```bash
   git clone <URL_DO_SEU_REPOSITORIO>
   cd hustvl-yolosmall
   ```

2. **Crie e ative um ambiente virtual:**
   ```bash
   # No Windows
   python -m venv .venv
   .venv\Scripts\activate

   # No macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

## Uso

Para executar o script de detecção de objetos, utilize o seguinte comando:

```bash
python src/main.py
```

O script irá:
1. Carregar a imagem de `images/avenida.jpg`.
2. Utilizar o modelo `hustvl/yolos-small` para detectar objetos na imagem.
3. Desenhar as caixas delimitadoras e os rótulos dos objetos detectados.
4. Exibir a imagem resultante.

## Estrutura do Projeto

```
.
├── .gitignore
├── README.md
├── images
│   └── avenida.jpg
├── requirements.txt
└── src
    └── main.py
```
