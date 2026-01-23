from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-vqa-base"
)

image = Image.open("cake2.JPG").convert("RGB")
question = "What do you see in this image?"

# ✅ IMPORTANT: question=question
inputs = processor(image, question=question, return_tensors="pt")

out = model.generate(**inputs)
answer = processor.decode(out[0], skip_special_tokens=True)

print("Answer:", answer)
