import gradio as gr
import requests
import torch
from gradio.media import get_image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

# Step 1: a pretrained ResNet-18 image classifier, switched to inference mode.
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()

# Step 2: human-readable labels for the 1,000 ImageNet classes.
response = requests.get("https://git.io/JJkYN")
labels = [line.strip() for line in response.text.split("\n") if line.strip()]

# The preprocessing ResNet expects: resize, center crop, tensor, ImageNet normalization.
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# predict maps the image to {class name: probability}; softmax turns logits into probabilities.
def predict(inp):
    inp = transform(inp).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
    return {labels[i]: float(prediction[i]) for i in range(len(labels))}


# Step 3: gr.Label(num_top_classes=3) shows the top 3 classes; examples prepopulate the demo.
gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    examples=[get_image("lion.jpg"), get_image("cheetah.jpg")],
).launch()
