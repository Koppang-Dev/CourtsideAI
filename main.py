from util import read_video, save_video
from tracker import Tracker
import os 
import cv2

def main():

    # Reading the video
    video_frames, fps = read_video('data/input_videos/lakerVideoSmall.mp4')

    # Tracker initalization
    tracker = Tracker('models/final_yolo_model.pt')

    # Run detection
    yolo_detections = tracker.get_yolo_detections(video_frames)

    # Draw only YOLO bounding boxes
    frames_with_boxes = tracker.draw_yolo_bboxes(video_frames, yolo_detections)

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs3.pkl')

    # Drawing output
    # Draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)


    # Saving video with only bounding boxes
    save_video(frames_with_boxes, 'output_videos/bounding_box.mp4', fps)


    # Saving video new ellipse
    save_video(output_video_frames, 'output_videos/output_video11.mp4', fps)




# Calling main
if __name__ == '__main__':
    main()