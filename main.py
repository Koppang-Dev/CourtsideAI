from util import read_video, save_video
from tracker import Tracker
import os 
import cv2

def main():

    # Reading the video
    video_frames, fps = read_video('input-videos/lakerVideoSmall.mp4')

    # Tracker initalization
    tracker = Tracker('models/yolo_models/final_yolo_model.pt', classification_model_path='models/classification_models/best_model2.pth')

    # Run detection
    yolo_detections = tracker.get_yolo_detections(video_frames)

    # Draw only YOLO bounding boxes
    frames_with_boxes = tracker.draw_yolo_bboxes(video_frames, yolo_detections)

    # Tracking using DeepSort
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=False, stub_path='stubs/track_stubs3.pkl')

    # Getting metrics for DeepSort
    metrics = tracker.visualize_deepsort_metrics(tracks)

    # Drawing output
    output_video_frames = tracker.draw_annotations(video_frames, tracks)


    # Saving video with only bounding boxes
    save_video(frames_with_boxes, 'output-videos/bounding_box.mp4', fps)

    # Saving final video with new ellipse
    save_video(output_video_frames, 'output-videos/output_video1.mp4', fps)

# Calling main
if __name__ == '__main__':
    main()