from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torch
import os

def main():
    
    img_path = os.path.join('images', 'avenida.jpg')
    img = Image.open(img_path)

    image_processor = YolosImageProcessor.from_pretrained('hustvl/yolos-small')
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-small')

    inputs = image_processor(images=img, return_tensors='pt')
    outputs = model(**inputs, )

    target_sizes = torch.tensor([img.size[::-1]])
    results = image_processor.post_process_object_detection(outputs=outputs, threshold=0.6, target_sizes=target_sizes)[0]

    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype('arial.ttf', size=20)
    except IOError:
        print("Fonte Arial não encontrada. A usar a fonte padrão.")
        font = ImageFont.load_default()

    for score, label, box in zip(results['scores'], results['labels'], results['boxes']):

        box = [round(i, 2) for i in box.tolist()]

        label_name = model.config.id2label[label.item()]
        
        draw.rectangle(box, outline='red', width=4)
        draw.text((box[0], box[1]),  f"{label_name} {round(score.item(), 2)}", fill='white', font=font)

    plt.figure(figsize=(10, 5))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()