import gradio as gr


def greet(name: str, is_morning: bool, temperature: float) -> tuple:
    salutation = "Good morning" if is_morning else "Hello"
    greetings = f"{salutation} {name}. It is {temperature} degrees today."
    celsius = (temperature - 32) * 5 / 9
    return greetings, round(celsius, 2)

with gr.Blocks() as demo:
    # Inputs
    name = gr.Textbox(label="First name")
    is_morning = gr.Checkbox(label="Is it morning?")
    temperature_fahreneit = gr.Number(label="Temperature (Fahrenheit)")
    greetings_button = gr.Button(label="Greet me")
    # Outputs
    greetings = gr.Textbox(label="Greetings")
    temperature_celsius = gr.Number(label="Temperature (Celsius)")

    # Connect inputs and outputs
    greetings_button.click(fn = greet, inputs= [name, is_morning, temperature_fahreneit], outputs=[greetings, temperature_celsius])
    
demo.launch()
