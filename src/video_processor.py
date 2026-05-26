import os
# 1. Skryjeme grafickou kartu před PyTorchem, aby nepadal starý ovladač
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import torch
# Importujeme pouze model builder
from sam3.model_builder import build_sam3_video_predictor

def initialize_model(device="cpu"):
    """Loads the SAM 3 video predictor and moves it to the target device."""
    print(f"Building SAM 3 video predictor...")
    # Sestavíme prediktor (zde se parametr device nezadává)
    video_predictor = build_sam3_video_predictor()
    
    # Přesuneme samotný model uvnitř prediktoru na CPU
    print(f"Moving model weights to device: '{device}'...")
    video_predictor.model.to(device)
    
    return video_predictor

def process_video(video_name, text_prompt):
    """Tracks objects in a local video using CPU."""
    input_path = os.path.join("data", "input_videos", video_name)
    output_path = os.path.join("data", "output_videos", f"tracked_{video_name}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found at: {input_path}. Please place your video there.")

    # Použijeme CPU
    device = "cpu"
    video_predictor = initialize_model(device)
    
    # 1. Spuštění relace
    print("Initializing video session (loading frames to RAM)...")
    response = video_predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=input_path,
        )
    )
    session_id = response["session_id"]
    
    # 2. Přidání textového promptu
    print(f"Adding text prompt: '{text_prompt}'...")
    response = video_predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=text_prompt,
        )
    )
    
    # Otevření videa pro uložení výsledku
    cap = cv2.VideoCapture(input_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("Tracking and generating output video (running on CPU)...")
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        annotated_frame = frame.copy()
        
        # Aplikace masek na aktuální snímek
        if "outputs" in response and "masks" in response["outputs"]:
            masks = response["outputs"]["masks"]
            for mask in masks:
                binary_mask = mask > 0.0
                annotated_frame[binary_mask] = [0, 255, 0] # Zelená maska
                
        out.write(annotated_frame)
        frame_idx += 1
        
        if frame_idx % 10 == 0:
            print(f"Processed frame {frame_idx}...")

    cap.release()
    out.release()
    print(f"Done! Tracked video saved to: {output_path}")

if __name__ == "__main__":
    VIDEO_NAME = "test.mp4" 
    PROMPT = "car" 
    
    try:
        process_video(VIDEO_NAME, PROMPT)
    except FileNotFoundError as e:
        print(e)
