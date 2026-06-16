import os
import gradio as gr
from huggingface_hub import InferenceClient
from PIL import Image

# Default code snippet to display when the app first loads
DEFAULT_CODE = """import os
from huggingface_hub import InferenceClient

# Initialize client with your API key
client = InferenceClient(
    provider="fal-ai",
    api_key="YOUR_HF_TOKEN",
)

# Generate image (returns a PIL.Image)
image = client.text_to_image(
    "Astronaut riding a horse",
    model="black-forest-labs/FLUX.1-dev",
)
"""

def generate_image(prompt, api_key):
    # Validate prompt
    if not prompt.strip():
        raise gr.Error("Error: Prompt cannot be empty!")
    
    # Check for API Key (fallback to environment variable HF_TOKEN)
    effective_api_key = api_key.strip()
    if not effective_api_key:
        effective_api_key = os.environ.get("HF_TOKEN", "").strip()
        if not effective_api_key:
            raise gr.Error("Error: Hugging Face API Token (HF_TOKEN) is required! Enter it in the textbox or set it as an environment variable.")

    try:
        # Initialize the Inference Client using fal-ai as the provider
        client = InferenceClient(
            provider="fal-ai",
            api_key=effective_api_key,
        )
        
        # Call the serverless Hugging Face API to generate the image
        image = client.text_to_image(
            prompt,
            model="black-forest-labs/FLUX.1-dev",
        )
        
        # Mask the API key in the displayed code for safety
        if len(effective_api_key) > 8:
            masked_key = f"{effective_api_key[:4]}...{effective_api_key[-4:]}"
        else:
            masked_key = "********"
            
        # Format the dynamic code output block showing the actual prompt used
        code_used = f"""import os
from huggingface_hub import InferenceClient

# Initialize client with the provided API key
client = InferenceClient(
    provider="fal-ai",
    api_key="{masked_key}",
)

# Generate image (returns a PIL.Image)
image = client.text_to_image(
    "{prompt.replace('"', '\\"')}",
    model="black-forest-labs/FLUX.1-dev",
)
"""
        return image, code_used
        
    except Exception as e:
        raise gr.Error(f"API Error: {str(e)}")

# Custom CSS for styling the UI
custom_css = """
.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}
.title-header {
    text-align: center;
    margin-bottom: 20px;
}
"""

# Create the Gradio interface
with gr.Blocks() as demo:
    # Use gr.Group to wrap components instead of Div
    with gr.Group(elem_classes="container"):
        gr.Markdown(
            """
            # 🎨 AI Image Generator
            ### Powered by Hugging Face Inference API & `FLUX.1-dev` (via fal-ai)
            """,
            elem_classes="title-header"
        )
        
        with gr.Row():
            # Left Column: Inputs
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Settings & Prompt")
                
                api_key_input = gr.Textbox(
                    label="Hugging Face API Token (HF_TOKEN)",
                    placeholder="Enter your hf_... token (or set HF_TOKEN env var)",
                    type="password",
                    value=os.environ.get("HF_TOKEN", ""),
                    info="Get your free token from huggingface.co/settings/tokens"
                )
                
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image you want to generate...",
                    lines=6,
                    value="Astronaut riding a horse"
                )
            
            # Right Column: Controls, Preview, and Code
            with gr.Column(scale=1):
                gr.Markdown("### 🚀 Controls & Output")
                
                generate_btn = gr.Button(
                    "✨ Generate Image", 
                    variant="primary",
                    size="lg"
                )
                
                # Image Preview Output
                image_output = gr.Image(
                    label="Generated Image Preview", 
                    type="pil",
                    interactive=False
                )
                
                # Code Display
                code_output = gr.Code(
                    label="Code Executed",
                    language="python",
                    value=DEFAULT_CODE,
                    interactive=False
                )
                
        # Link action to execute on generate button click
        generate_btn.click(
            fn=generate_image,
            inputs=[prompt_input, api_key_input],
            outputs=[image_output, code_output]
        )

# Run the app, passing theme and css to launch() for Gradio 6.0 compatibility
if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(primary_hue="purple", secondary_hue="indigo"),
        css=custom_css
    )
