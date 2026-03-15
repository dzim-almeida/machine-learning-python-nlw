from google import genai
from PIL import Image
import os

client = genai.Client(api_key="AIzaSyCvW458zpcLtLKDRmVXRPls5676awk2A8k")

def main():

    try:
        image_path = os.path.join("images", "plane-001.jpeg")
        image = Image.open(image_path)
    except Exception as e:
        print(f"Erro ao carregar a imagem: {e}")
        return
    
    prompt = [
        "Classifique o objeto na imagem e forneça uma descrição detalhada.",
        image
    ]

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    print(response.text)

if __name__ == "__main__":
    main()
