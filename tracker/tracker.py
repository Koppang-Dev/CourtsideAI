import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import pickle
import sys
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision import transforms, models
import matplotlib.pyplot as plt
from collections import defaultdict


sys.path.append('../')
from util import get_center_of_bbox, get_bbox_width

class Tracker:
    def __init__(self, model_path, classification_model_path):
        
        # Mapping Class Indicies to Player names
        self.idx_to_class = {
            0: 'Dorian Finney-Smith',
            1: 'Gabe Vincent',
            2: 'Ivica Zubac',
            3: 'James Harden',
            4: 'Jaxson Hayes',
            5: 'Kawhi Leonard',
            6: 'Kris Dunn',
            7: 'LeBron James',
            8: 'Luka Doncic',
            9: 'Norman Powell'
        }

        # Tracking variables for unique player assignments
        self.assigned_classes = set()  
        self.track_history = {}        
        self.min_frames_for_assignment = 5  
        self.reassignment_threshold = 0.2  
        self.confidence_threshold = 0.7    

        # Initializing YOLO model
        self.model = YOLO(model_path)
        self.model.model.eval()

        
        # Initializing Deep SORT tracker
        self.deepsort = DeepSort()

        # Initializing ResNet18 model architecture
        self.classification_model = models.resnet50(pretrained=False)  # Initialize the model without pretrained weights

        # Modify the final fully connected layer (adjust according to your dataset)
        num_classes = 10 # Update this with your actual number of classes
        self.classification_model.fc = nn.Sequential(
                nn.Linear(self.classification_model.fc.in_features, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.7),  
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            
            # These are the original best_model parameters
            # nn.Dropout(0.3),
            # nn.Linear(self.classification_model.fc.in_features, 512),
            # nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(512, num_classes)
        )

        # Load the saved weights into the model
        try:
            self.classification_model.load_state_dict(torch.load(classification_model_path, map_location=torch.device('cpu')))
            print("Classification model weights loaded successfully.")
        except Exception as e:
            print(f"Error loading classification model weights: {e}")
            raise

        # Set model to evaluation mode
        self.classification_model.eval()


         # Define the image transformation pipeline
        self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),  
                transforms.CenterCrop(224),     
                transforms.ColorJitter(brightness=0.2, contrast=0.2),  
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
   

    def update_player_assignment(self, track_id, new_class_idx, confidence):
        """Strictly enforce unique class assignments based on highest confidence"""
        # Initialize track history if needed
        if track_id not in self.track_history:
            self.track_history[track_id] = {
                "class": None,
                "confidence": 0.0,
                "frames": 0
            }
        
        history = self.track_history[track_id]
        history["frames"] += 1
        
        # Case 1: This track already has an assigned class
        if history["class"] is not None:
            # If this detection matches our current class, update confidence if higher
            if new_class_idx == history["class"]:
                if confidence > history["confidence"]:
                    history["confidence"] = confidence
            # If this is a different class, only consider switching if:
            # 1. The new class isn't already assigned to someone else, AND
            # 2. Our confidence is higher than the current assignment's confidence
            elif (new_class_idx not in self.assigned_classes and 
                confidence > self.confidence_threshold and
                (confidence > history["confidence"] * (1 + self.reassignment_threshold))):
                # Release our current class
                if history["class"] in self.assigned_classes:
                    self.assigned_classes.remove(history["class"])
                # Take the new class
                self.assigned_classes.add(new_class_idx)
                history["class"] = new_class_idx
                history["confidence"] = confidence
        
        # Case 2: This track has no assigned class yet
        else:
            # Only assign if:
            # 1. The class isn't already taken, AND
            # 2. We meet the confidence threshold
            if (new_class_idx not in self.assigned_classes and 
                confidence >= self.confidence_threshold):
                self.assigned_classes.add(new_class_idx)
                history["class"] = new_class_idx
                history["confidence"] = confidence
        
        return history["class"]


    # Image Classification with Player's Name
    def classify_with_trained_model(self, cropped_image):
        # Convert to PyTorch tensor and preprocess
        cropped_image = self.transform(cropped_image).unsqueeze(0)  # Add batch dimension

        # Make the prediction for the player's name
        with torch.no_grad():
            outputs = self.classification_model(cropped_image)
            confidence_score, predicted_class = torch.max(outputs, 1)

        return predicted_class.item(), confidence_score.item()

    # Object Detection With Yolo
    def get_yolo_detections(self, frames, conf_threshold=0.3):
        all_detections = []
        for frame in frames:
            with torch.no_grad():
                results = self.model.predict(source=frame, conf=conf_threshold, save=False, verbose=False)[0]

            frame_detections = []


            for det in results.boxes:
                bbox = det.xyxy[0].tolist()  # [x1, y1, x2, y2]
                conf = det.conf[0].item()
                cls_id = int(det.cls[0].item())
                class_name = results.names[cls_id]

                print(f"Detected class id: {cls_id}, class name: {class_name}")

                frame_detections.append({
                    "bbox": bbox,
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": results.names[cls_id]
                })

            all_detections.append(frame_detections)
        return all_detections

    # Drawing Bounding Boxes
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

    # Detecting Frames From Video
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections.extend(detections_batch)
        return detections

    # Tracking Using DeepSort
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
            detections = []

            for det in results.boxes:
                bbox = det.xyxy[0].tolist()  # [x1, y1, x2, y2]
                conf = det.conf[0].item()
                cls_id = int(det.cls[0].item())
                confidences.append(conf)
                class_ids.append(cls_id)
                
                # Convert bbox to [left, top, width, height]
                left = bbox[0]
                top = bbox[1]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]

                # Append in the required format: [[left, top, width, height], confidence, class_id]
                detections.append([[left, top, width, height], conf, cls_id])


            # Update Deep SORT
            tracks_deep_sort = self.deepsort.update_tracks(detections,frame=frame)

            # Tracks dictionary for each class
            tracks["person"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            # Processing each track
            for track in tracks_deep_sort:
                if not track.is_confirmed():
                    continue
                    
                track_id = track.track_id
                bbox = track.to_tlbr()

                if track.det_class == cls_names_inv.get("person", -1):
                    tracks["person"][frame_num][track_id] = {"bbox": bbox}
                elif track.det_class == cls_names_inv.get("ball", -1):
                    tracks["ball"][frame_num][track_id] = {"bbox": bbox}
                elif track.det_class == cls_names_inv.get("referee", -1):
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, colour, track_id):
        y2 = int(bbox[3])
        x_ceneter, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_ceneter, y2),
            axes=((int(width), int(0.35 * width))),
            angle=0.0,
            startAngle=45,
            endAngle=235,
            color=colour,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        return frame

    def draw_annotations(self, video_frames, tracks, output_dir="cropped_images"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks["person"][frame_num] if frame_num < len(tracks["person"]) else {}

            # Clean up departed tracks from assigned_classes
            current_tracks = set(player_dict.keys())
            for track_id in list(self.track_history.keys()):
                if track_id not in current_tracks:
                    if self.track_history[track_id]["class"] is not None:
                        self.assigned_classes.discard(self.track_history[track_id]["class"])
                    del self.track_history[track_id]

            for track_id, player in player_dict.items():
                bbox = player["bbox"]
                x1, y1, x2, y2 = map(int, bbox)

                # Skip invalid bounding boxes
                if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
                    continue  
                    
                # Padding around the bounding box (20% of width/height)
                padding_w = int(0.2 * (x2 - x1))
                padding_h = int(0.2 * (y2 - y1))
                x1_pad = max(0, x1 - padding_w)
                y1_pad = max(0, y1 - padding_h)
                x2_pad = min(frame.shape[1] - 1, x2 + padding_w)
                y2_pad = min(frame.shape[0] - 1, y2 + padding_h)

                # Cropped Player
                cropped_player = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                    
                if (cropped_player is None or cropped_player.size == 0 or 
                    cropped_player.shape[0] < 20 or cropped_player.shape[1] < 20):
                    continue
            
                # Enhancing Cropped Image
                lab = cv2.cvtColor(cropped_player, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE for better contrast
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                enhanced_lab = cv2.merge((cl, a, b))
                enhanced_crop = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

                # Apply a sharpening filter
                sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                enhanced_crop = cv2.filter2D(enhanced_crop, -1, sharpen_kernel)

                # Optional: Upscale image using high-quality interpolation
                upscale_factor = 2
                enhanced_crop = cv2.resize(enhanced_crop, None, fx=upscale_factor, fy=upscale_factor, 
                                         interpolation=cv2.INTER_LANCZOS4)

               # Classify the enhanced crop
                predicted_class_idx, confidence_score = self.classify_with_trained_model(enhanced_crop)

                # Update tracking with uniqueness enforcement
                assigned_class = self.update_player_assignment(track_id, predicted_class_idx, confidence_score)

                # Determine what to display
                display_class_idx = assigned_class if assigned_class is not None else predicted_class_idx
                predicted_class_name = self.idx_to_class.get(display_class_idx, "Unknown Player")

                # Always show just the player name
                display_name = predicted_class_name
                is_confident = assigned_class is not None and confidence_score >= self.confidence_threshold

                # Save with tracking info (now without Player X prefix)
                safe_name = "".join([c if c.isalnum() else "_" for c in predicted_class_name])
                filename = f"frame_{frame_num}_track_{track_id}_{safe_name}.jpg"
                # cv2.imwrite(os.path.join(output_dir, filename), enhanced_crop)

                # Draw annotations with confidence indicator
                if not is_confident:
                    frame = self.draw_ellipse(frame, bbox, (0, 165, 255), track_id)  # Orange for uncertain
                else:
                    frame = self.draw_ellipse(frame, bbox, (0, 0, 255), track_id)  # Red for confident

                cv2.putText(frame, display_name, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            output_video_frames.append(frame) 
        return output_video_frames
    
    def visualize_deepsort_metrics(self, tracks, output_dir="deepsort_metrics"):
        """Generate and save DeepSORT tracking metrics visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize metrics storage
        frame_numbers = []
        active_tracks = []
        id_switches = []
        track_lifetimes = defaultdict(int)
        current_ids = set()
        
        # Process track history to calculate metrics
        for frame_num, frame_tracks in enumerate(tracks["person"]):
            frame_numbers.append(frame_num)
            active_tracks.append(len(frame_tracks))
            
            # Track ID switches
            new_ids = set(frame_tracks.keys())
            switches = len(current_ids - new_ids)  # IDs that disappeared
            id_switches.append(switches)
            current_ids = new_ids
            
            # Update track lifetimes
            for track_id in frame_tracks:
                track_lifetimes[track_id] += 1
        
        # 1. Active Tracks Over Time
        plt.figure(figsize=(12, 6))
        plt.plot(frame_numbers, active_tracks, color='royalblue')
        plt.title('Active Player Tracks Per Frame')
        plt.xlabel('Frame Number')
        plt.ylabel('Number of Active Tracks')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'active_tracks.png'))
        plt.close()
        
        # 2. ID Switches Over Time
        plt.figure(figsize=(12, 6))
        plt.plot(frame_numbers, id_switches, color='crimson')
        plt.title('ID Switches Per Frame')
        plt.xlabel('Frame Number')
        plt.ylabel('Number of ID Switches')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'id_switches.png'))
        plt.close()
        
        # 3. Track Lifetime Distribution
        plt.figure(figsize=(12, 6))
        plt.hist(track_lifetimes.values(), bins=20, color='seagreen')
        plt.title('Distribution of Track Lifetimes')
        plt.xlabel('Frames Tracked')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'track_lifetimes.png'))
        plt.close()
        
        # 4. Tracking Summary Metrics
        summary_metrics = {
            'Total Tracks': len(track_lifetimes),
            'Avg Track Lifetime': np.mean(list(track_lifetimes.values())),
            'Total ID Switches': sum(id_switches),
            'Max Active Tracks': max(active_tracks)
        }
        
        with open(os.path.join(output_dir, 'metrics_summary.txt'), 'w') as f:
            for metric, value in summary_metrics.items():
                f.write(f"{metric}: {value}\n")
        
        return summary_metrics