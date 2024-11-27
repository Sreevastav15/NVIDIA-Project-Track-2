import gradio as gr
import moviepy.editor as mp
import matplotlib.pyplot as plt
import numpy as np
from vlm import VLM  # Assuming `vlm.py` is in the same directory

# VLM API Setup
def vlm_callback(message, reply, **kwargs):
    print("Callback message:", message)
    print("VLM response:", reply)

vlm = VLM(
    url="https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-11b-vision-instruct",
    api_key="ZGl2Z3NsdnFscjI0ZGc3aWs1c2c0Mjh2b2w6YWMyNWU4YWEtM2I4Ny00MTA0LThjMmEtMTljNGFlM2RlYWE3",  # Replace with your NVIDIA API key
    callback=vlm_callback,
)

# Function to split videos into 5 parts
def split_video(video_path, parts=5):
    video = mp.VideoFileClip(video_path)
    part_duration = video.duration / parts
    parts_paths = []

    for i in range(parts):
        start_time = i * part_duration
        end_time = (i + 1) * part_duration
        trimmed_part = video.subclip(start_time, end_time)
        part_path = f"part_{i + 1}.mp4"
        trimmed_part.write_videofile(part_path, codec="libx264")
        parts_paths.append(part_path)

    return parts_paths

# Simulate API Call for Recognition
def get_recognition_rate(video_path):
    # Placeholder: Replace with actual VLM API frame-by-frame processing
    return np.random.randint(70, 100)  # Simulate random recognition rates for each part

# Process videos and generate line graph
def process_videos_and_plot(video1_path, video2_path, action, trim_length):
    # Split videos into 5 parts
    video1_parts = split_video(video1_path)
    video2_parts = split_video(video2_path)

    # Collect recognition rates
    video1_rates = [get_recognition_rate(part) for part in video1_parts]
    video2_rates = [get_recognition_rate(part) for part in video2_parts]

    # Create line graph
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 6), video1_rates, marker='o', label="Video 1", color="blue")
    plt.plot(range(1, 6), video2_rates, marker='o', label="Video 2", color="green")

    plt.title("Recognition Rates Over Time")
    plt.xlabel("Video Frame")
    plt.ylabel("Recognition Rate (%)")
    plt.ylim(0, 100)
    plt.xticks(range(1, 6))
    plt.legend()
    plt.grid(alpha=0.5)

    # Save plot to file
    plot_path = "recognition_rates_line_graph.png"
    plt.savefig(plot_path)
    plt.close()

    return plot_path

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("### Real-Time Recognition and Visualization")

    video1_input = gr.Video(label="Input Video 1")
    video2_input = gr.Video(label="Input Video 2")
    trim_length_input = gr.Number(label="Trim Length (seconds)", value=20)
    action_input = gr.Textbox(label="Action to Detect (e.g., jumping, running)")
    graph_output = gr.Image(label="Recognition Rates Line Graph")

    submit_btn = gr.Button("Process Videos and Generate Graph")
    submit_btn.click(
        process_videos_and_plot,
        inputs=[video1_input, video2_input, action_input, trim_length_input],
        outputs=[graph_output],
    )

demo.launch()
