import cv2 as cv
import os

def read_video(video_path):
    cap = cv.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release() 
    return frames


def save_video(output_video_frames, output_video_path):

    if not os.path.exists(output_video_path):
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    
    out = cv.VideoWriter(
        output_video_path,
        fourcc,
        24,
        (output_video_frames[0].shape[1], output_video_frames[0].shape[0])
    )

    for frame in output_video_frames:
        out.write(frame)
    out.release()