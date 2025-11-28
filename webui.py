import sys
import argparse
parser = argparse.ArgumentParser(description="FlashVSR WebUI")
parser.add_argument("--listen", action="store_true", help="Allow LAN access")
parser.add_argument("--port", type=int, default=7860, help="Service Port")
if __name__ == "__main__":
    args = parser.parse_args()
else:
    args = parser.parse_args([])

import gradio as gr
import os
import shutil

from flashvsr_runner import run_flashvsr_integrated

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
TEMP_DIR = os.path.join(ROOT_DIR, "_temp")
os.environ['GRADIO_TEMP_DIR'] = TEMP_DIR


def create_ui():
    with gr.Blocks(theme=gr.themes.Soft(), title="FlashVSR+ WebUI") as demo:
        gr.Markdown("### FlashVSR+ WebUI")
        
        with gr.Row():
            with gr.Column(scale=1):
                input_video = gr.Video(label="Upload Video File", height=412)
                run_button = gr.Button("Start Processing", variant="primary")
                gr.Markdown("### Main Settings")
                with gr.Group():
                    with gr.Row():
                        mode_radio = gr.Radio(choices=["tiny", "tiny-long", "full"], value="tiny", label="Pipeline Mode")
                        seed_number = gr.Number(value=-1, label="Seed", precision=0)
                with gr.Group():
                    with gr.Row():
                        scale_slider = gr.Slider(minimum=2, maximum=4, step=1, value=4, label="Upscale Factor")
                        tiled_dit_checkbox = gr.Checkbox(label="Enable Tiled DiT", info="For very high-resolution videos or low VRAM scenarios", value=False)
                    with gr.Row(visible=False) as tiled_dit_options:
                        tile_size_slider = gr.Slider(minimum=64, maximum=512, step=16, value=256, label="Tile Size")
                        tile_overlap_slider = gr.Slider(minimum=8, maximum=128, step=8, value=24, label="Tile Overlap")
                        
            with gr.Column(scale=1):
                video_output = gr.Video(label="Output Result", interactive=False, height=620)
                gr.Markdown("### Advanced Settings")
                with gr.Accordion("Expand Advanced Options", open=False):
                    model_version = gr.Radio(choices=["FlashVSR", "FlashVSR-v1.1"], value="FlashVSR-v1.1", label="Model Version")
                    sparse_ratio_slider = gr.Slider(minimum=0.5, maximum=5.0, step=0.1, value=2.0, label="Sparse Ratio", info="Controls attention sparsity; smaller values are more sparse")
                    kv_ratio_slider = gr.Slider(minimum=1, maximum=8, step=1, value=3, label="KV Cache Ratio", info="Controls the length of the KV cache")
                    local_range_slider = gr.Slider(minimum=3, maximum=15, step=2, value=11, label="Local Range", info="Size of the local attention window")
                    attention_mode_radio = gr.Radio(choices=["sage", "block"], value="sage", label="Attention Mode")
                    color_fix_checkbox = gr.Checkbox(label="Enable Color Fix", value=True)
                    tiled_vae_checkbox = gr.Checkbox(label="Enable Tiled VAE", value=True)
                    unload_dit_checkbox = gr.Checkbox(label="Unload DiT before decoding (saves VRAM)", value=False)
                    dtype_radio = gr.Radio(choices=["fp16", "bf16"], value="bf16", label="Data Type")
                    device_textbox = gr.Textbox(value="auto", label="Device", info="e.g., 'auto', 'cuda:0', 'cpu'")
                    quality_slider = gr.Slider(minimum=1, maximum=10, step=1, value=6, label="Output Video Quality")
                    fps_number = gr.Number(value=30, label="Output FPS (for image sequence input only)", precision=0)
                    
        def toggle_tiled_dit_options(is_checked):
            return gr.update(visible=is_checked)
        
        tiled_dit_checkbox.change(fn=toggle_tiled_dit_options, inputs=[tiled_dit_checkbox], outputs=[tiled_dit_options])
        
        run_button.click(
            fn=run_flashvsr_integrated,
            inputs=[
                input_video, model_version, mode_radio, scale_slider, color_fix_checkbox, tiled_vae_checkbox, 
                tiled_dit_checkbox, tile_size_slider, tile_overlap_slider, unload_dit_checkbox, 
                dtype_radio, seed_number, device_textbox, fps_number, quality_slider, attention_mode_radio,
                sparse_ratio_slider, kv_ratio_slider, local_range_slider # Added new parameters
            ],
            outputs=[video_output]
        )
        
    return demo

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)

    ui = create_ui()
    if args.listen:
        ui.queue().launch(share=False, server_name="0.0.0.0", server_port=args.port)
    else:
        ui.queue().launch(share=False, server_port=args.port)