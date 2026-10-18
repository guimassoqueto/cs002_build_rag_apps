import gradio as gr
from transformers import BlipForConditionalGeneration, BlipProcessor

from app.config.envs import BLIP_MODEL


# A first Gradio app: inputs=["text", "slider"] are shortcuts for gr.Textbox() and gr.Slider().
def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)


greet_demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)

# BLIP (Bootstrapped Language Image Pretraining) generates a caption for an image.
processor = BlipProcessor.from_pretrained(BLIP_MODEL)
model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL)


def caption_image(image):
    """Takes a PIL Image input and returns a caption."""
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs)
    return processor.decode(outputs[0], skip_special_tokens=True)


# gr.Image(type="pil") hands the function a PIL image; title and description add context.
caption_demo = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Image Captioning with BLIP",
    description="Upload an image to generate a caption.",
)

# launch() serves the demos on http://127.0.0.1:7860.
gr.TabbedInterface(
    [greet_demo, caption_demo],
    ["Greeting", "Image captioning"],
).launch(server_name="127.0.0.1", server_port=7860)
