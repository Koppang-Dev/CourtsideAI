import cv2
import os

# Reads in a video
# Returns the list of frames for the video
def read_video(video_path):

    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(f"[ERROR] Video file does not exist or path is invalid: {video_path}")

    # Video Capture object
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)


    # Collecting the frames
    frames = []
    while True:
        ret, frame = cap.read()

        # If ret = False then the video has ended
        if not ret:
            break
            
        # Collected list of frames for the video
        frames.append(frame)
    
    return frames, fps 


# Saving the frames to a selected video path
def save_video(output_video_frames, output_video_path, fps):

    # Defining the output format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Defining the video writer 
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))

    # Writing each frame to the video writer
    for frame in output_video_frames:
        out.write(frame)
    out.release()
