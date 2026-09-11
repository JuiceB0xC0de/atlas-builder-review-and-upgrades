import gradio as gr

def status():
    return (
        "Atlas runner Space is alive.\n"
        "This is a headless pipeline — open the dev-mode terminal and run:\n"
        "  python app.py --corpus prompts/prompts.jsonl ..."
    )

demo = gr.Interface(fn=status, inputs=None, outputs="text",
                    title="atlasing — dev-mode runner")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)