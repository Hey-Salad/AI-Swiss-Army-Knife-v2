import gradio as gr
import torch
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
import trimesh
import os
import logging
import time
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('server.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
PIPELINE_CONFIG = {
    'num_inference_steps': 25,  # Reduced from 50 for faster processing
    'guidance_scale': 7.5,
    'border_ratio': 0.1  # Reduced from 0.15 for tighter mesh
}

class Model3DGenerator:
    def __init__(self):
        self.model = None
        self.initialize_model()

    def initialize_model(self):
        try:
            logger.info("Loading Hunyuan3D model...")
            self.model = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
            self.model.to('cuda')
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def preprocess_image(self, image):
        """Preprocess image for better 3D generation"""
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to smaller size for faster processing
            target_size = (384, 384)
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            
            return image
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise

    def generate_3d(self, image, progress=gr.Progress()):
        """Generate 3D model from image"""
        try:
            start_time = time.time()
            logger.info("Starting generation process")

            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Generate mesh with optimized settings
            logger.info("Generating 3D mesh...")
            progress(0.2, "Generating 3D mesh...")
            
            mesh = self.model(
                image=processed_image,
                num_inference_steps=PIPELINE_CONFIG['num_inference_steps'],
                guidance_scale=PIPELINE_CONFIG['guidance_scale'],
                border_ratio=PIPELINE_CONFIG['border_ratio']
            )[0]

            progress(0.8, "Exporting model...")
            
            # Save to temporary GLB file
            temp_file = "temp_model.glb"
            mesh.export(temp_file)

            end_time = time.time()
            processing_time = end_time - start_time
            logger.info(f"Generation completed in {processing_time:.2f} seconds")

            progress(1.0, "Complete!")
            return temp_file, f"Processing time: {processing_time:.2f} seconds"

        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

def create_ui():
    generator = Model3DGenerator()
    
    with gr.Blocks(title="3D Model Generator") as app:
        gr.Markdown("""
        # 3D Model Generator
        Upload an image to generate a 3D model. Best results with:
        - Single object
        - Plain background
        - Good lighting
        - Clear view of the object
        
        Note: Currently generating shape only (without texture) due to GPU VRAM limitations.
        """)
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="numpy")
                generate_button = gr.Button("Generate 3D Model")
            
            with gr.Column():
                output_model = gr.File(label="Generated 3D Model (GLB)")
                status_text = gr.Textbox(label="Status", interactive=False)

        # Handle generation
        generate_button.click(
            fn=generator.generate_3d,
            inputs=[input_image],
            outputs=[output_model, status_text]
        )
        
        gr.Markdown("""
        ### System Information
        - GPU: """ + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected") + """
        - Settings:
          - Inference steps: """ + str(PIPELINE_CONFIG['num_inference_steps']) + """
          - Guidance scale: """ + str(PIPELINE_CONFIG['guidance_scale']) + """
        """)

    return app

if __name__ == "__main__":
    # Create and launch the UI
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=8083,
        share=True  # Creates a public URL
    )
