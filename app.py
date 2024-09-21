import gradio as gr
import numpy as np 
import torch 
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
import logging
import os

logging.basicConfig(level=logging.DEBUG)

# Set device to CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Define the path to the checkpoint file
sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
if not os.path.exists(sam_checkpoint):
    logging.error(f"Checkpoint file not found: {sam_checkpoint}")
    exit(1)

model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

# Load Stable Diffusion Inpainting Pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
pipe = pipe.to(device)

selected_pixels = []

with gr.Blocks() as demo:
    with gr.Row():
        input_img = gr.Image(label="Input")
        mask_img = gr.Image(label="Mask")
        output_img = gr.Image(label="Output")

    with gr.Row():
        prompt_text = gr.Textbox(lines=1, label="Prompt")

    with gr.Row():
        submit = gr.Button("Submit")

    def generate_mask(image, evt: gr.SelectData):
        logging.debug("Generating mask...")
        selected_pixels.append(evt.index)

        predictor.set_image(np.array(image))
        input_points = np.array(selected_pixels)
        input_labels = np.ones(input_points.shape[0])
        masks, _, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False
        )
        mask = Image.fromarray(masks[0, :, :])
        return mask
    
    def inpaint(image, mask, prompt):
        logging.debug("Inpainting...")
        image = Image.fromarray(np.array(image))
        mask = Image.fromarray(np.array(mask))

        image = image.resize((512, 512))
        mask = mask.resize((512, 512))

        output = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask
        ).images[0]
        
        return output
    
    input_img.select(generate_mask, [input_img], [mask_img])

    submit.click(
        inpaint,
        inputs=[input_img, mask_img, prompt_text],
        outputs=[output_img]
    )

if __name__ == "__main__":
    logging.debug("Launching Gradio app...")
    demo.launch()