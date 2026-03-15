from google import genai
from PIL import Image
import os

client = genai.Client(api_key="your-api-key-here")

def main():

    try:
        image_path = os.path.join("your-image-here")
        image = Image.open(image_path)
    except Exception as e:
        print(f"Erro ao carregar a imagem: {e}")
        return
    
    prompt = [
        "your-question-here",
        image
    ]

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    print(response.text)

if __name__ == "__main__":
    main()
