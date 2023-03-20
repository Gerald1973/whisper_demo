import gradio as gr


def greet(name: str, is_morning: bool, temperature: float) -> tuple:
    salutation = "Good morning" if is_morning else "Hello"
    greetings = f"{salutation} {name}. It is {temperature} degrees today."
    celsius = (temperature - 32) * 5 / 9
    return greetings, round(celsius, 2)


demo = gr.Interface(
    fn=greet, inputs=["text", "checkbox", "number"], outputs=["text", "number"])

demo.launch()
