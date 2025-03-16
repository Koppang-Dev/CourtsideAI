from ultralytics import YOLO
import cv2
import supervision as sv
import pickle
import os
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

sys.path.append('../')
from util import get_center_of_bbox, get_bbox_width

class Tracker:
    def __init__(self, model_path):

        # Initalizing model
        self.model = YOLO(model_path)

         # Initalizing Deep SORT tracker
        self.deepsort = DeepSort()

        # Defining the VGG-16 model for player classification
        self.vgg_model = load_model('models/lakers_players_model.h5')
    
    # Classifying player frame with Vgg-16 model
    def classify_with_vgg16(self, cropped_image):

        # Resizing the image
        cropped_image = cv2.resize(cropped_image, (224, 224))

        # Processing the image for VGG-16
        img_array = image.img_to_array(cropped_image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Prediction with the VGG-16 model
        predictions = self.vgg_model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)

        # Returning the predicted label
        return predicted_class[0]
    
    def get_yolo_detections(self, frames, conf_threshold=0.1):
        all_detections = []

        for frame in frames:
            results = self.model.predict(source=frame, conf=conf_threshold, save=False, verbose=False)[0]

            frame_detections = []

            for det in results.boxes:
                bbox = det.xyxy[0].tolist()  # [x1, y1, x2, y2]
                conf = det.conf[0].item()
                cls_id = int(det.cls[0].item())

                frame_detections.append({
                    "bbox": bbox,
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": results.names[cls_id]
                })

            all_detections.append(frame_detections)

        return all_detections

    def draw_yolo_bboxes(self, frames, detections):
        output_video_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            for det in detections[frame_num]:
                bbox = list(map(int, det['bbox']))
                class_name = det['class_name']
                confidence = det['confidence']

                # Draw the rectangle
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                # Add label text
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            output_video_frames.append(frame)

        return output_video_frames



    def detect_frames(self, frames):

        # Adding a batch size to avoid memory overflow
        batch_size = 20

        # Detections list
        detections = []

        # Going through the batch sizes
        # Predicting on batch sizes
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch

        return detections
    

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        tracks = {
            "person": [],
            "referees": [],
            "ball": [],
        }

        for frame_num, frame in enumerate(frames):
            # Detect on single frame
            results = self.model.predict(source=frame, conf=0.1, save=False, verbose=False)[0]
            cls_names = results.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert to Deep SORT input
            bboxes = []
            confidences = []
            class_ids = []

            for det in results.boxes:
                bbox = det.xyxy[0].tolist()  # [x1, y1, x2, y2]
                conf = det.conf[0].item()
                cls_id = int(det.cls[0].item())

                bboxes.append(bbox)
                confidences.append(conf)
                class_ids.append(cls_id)

            # Update Deep SORT
            tracks_deep_sort = self.deepsort.update_tracks(bboxes, confidences, class_ids, frame=frame)

            tracks["person"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for track in tracks_deep_sort:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                bbox = track.to_tlbr()
                cls_id = track.cls

                if cls_id == cls_names_inv["person"]:
                    tracks["person"][frame_num][track_id] = {"bbox": bbox}
                elif cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][track_id] = {"bbox": bbox}
                elif cls_id == cls_names_inv.get("referee", -1):
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks


    
    # Draws the ellipse underneath the object
    def draw_ellipse(self, frame, bbox, colour, track_id):

        # y2 is the bottom of the boundary box
        y2 = int(bbox[3])

        # Getting bounding box center
        x_ceneter, _ = get_center_of_bbox(bbox)

        # Getting bounding box width
        width = get_bbox_width(bbox)

        # Creating eclipse
        cv2.ellipse(
            frame, # Current frame
            center=(x_ceneter, y2), # X center and Y center
            axes=((int(width), int(0.35 * width))), # Determines the shape based on minor axes
            angle=0.0,
            startAngle=45,
            endAngle=235,
            color=colour,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        # Return new bbox frame
        return frame





    # Drawing a circle beneath the user
    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []

        # Iterate through the frames
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            # Dictionary for our 3 objects
            player_dict = tracks["person"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]
        
            # Drawing the person
            for track_id , player in player_dict.items():

                # Get the bounding box
                bbox = player["bbox"]
                x1, y1, x2, y2 = map(int, bbox)

                # Crop the detected player from the frame
                cropped_player = frame[y1:y2, x1:x2]

                # Classify the cropped player using VGG-16
                predicted_class = self.classify_with_vgg16(cropped_player)

                # Draw an ellipse with the color of red
                frame = self.draw_ellipse(frame, bbox, (0, 0, 255), track_id)

                # Add classification label to the frame
                label = f"Player {track_id}: Class {predicted_class}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Adding the new frame
            output_video_frames.append(frame)
        
        return output_video_frames

















