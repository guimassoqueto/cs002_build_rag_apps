import gradio as gr


# A Gradio app starts as a plain Python function holding the logic.
def greet(name):
    return f"Hello, {name}!"


# gr.Interface wires the function to input and output components (here gr.Textbox).
text_demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(label="Name"),
    outputs=gr.Textbox(label="Greeting"),
)


# Multiple inputs are a list of components, passed to the function in order.
def repeat(text, times):
    return " ".join([text] * int(times))


multi_input_demo = gr.Interface(
    fn=repeat,
    inputs=[gr.Textbox(label="Text"), gr.Number(label="Times", value=3)],
    outputs=gr.Textbox(label="Result"),
)


# gr.File with file_count="multiple" uploads several files; the function receives their paths.
def count_files(files):
    return f"{len(files or [])} file(s) uploaded"


upload_demo = gr.Interface(
    fn=count_files,
    inputs=gr.File(file_count="multiple", label="Files"),
    outputs=gr.Textbox(label="Count"),
)

# launch() starts a local server and prints the URL to open in the browser.
gr.TabbedInterface(
    [text_demo, multi_input_demo, upload_demo],
    ["Text", "Multiple inputs", "File upload"],
).launch()
