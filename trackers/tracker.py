from ultralytics import YOLO
import supervision as sv
import pickle
import os
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.bbox_utils import get_center_of_bbox , get_bbox_width, get_foot_position
import cv2 as cv
import numpy as np

class Tracker:

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def draw_triangles(self , frame ,bbox, color) :
        y = int(bbox[1])
        x , _ = get_center_of_bbox(bbox)

        triangle_points = np.array([[x,y], [x-10, y-20], [x+10, y-20]], np.int32)

        cv.drawContours(frame , [triangle_points] , 0 , color , cv.FILLED)
        cv.drawContours(frame , [triangle_points] , 0 , (0,0,0) , 2)

        return frame
    
    def add_position_to_tracks(self , tracks):
        for object , object_tracks in tracks.items():
            for frame_num , track in enumerate(object_tracks):
                for track_id , track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    
                    tracks[object][frame_num][track_id]['position'] = position

    def interploate_ball_position(self , ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]

        df_ball_positions = pd.DataFrame(ball_positions , columns=['x1','y1','x2','y2'])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        return  [{1: {"bbox" : x}} for x in df_ball_positions.to_numpy().tolist()]
         
    def detect_frames(self, frames):
        batch_size=20

        detections = []
        for i in range(0, len(frames), batch_size):
            detection_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detection_batch
            
        return detections
    
    def draw_ellipse(self , frame , bbox , color , track_id=None):
        y2 = int(bbox[3])
        x_center , _ = get_center_of_bbox(bbox=bbox)
        width = get_bbox_width(bbox=bbox)

        cv.ellipse(
            frame, 
            center=(int(x_center), int(y2)), 
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=15,
            endAngle=235,         
            color=color,         
            thickness=2,
            lineType=cv.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20

        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2

        y1_rect = (y2 - rectangle_height //2) + 15
        y2_rect = (y2 + rectangle_height //2) + 15

        if track_id is not None:
            cv.rectangle(
                frame , 
                (int(x1_rect) , int(y1_rect)),
                (int(x2_rect) , int(y2_rect)),
                color, 
                cv.FILLED
            )

            x1_text = x1_rect+12

            if track_id > 99:
                x1_text -=10
            
            cv.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def get_object_tracks(self, frames , read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None:
            if not os.path.exists(os.path.dirname(stub_path)):
                os.makedirs(os.path.dirname(stub_path), exist_ok=True)

            if os.path.exists(stub_path):
                with open(stub_path, 'rb') as f:
                    tracks = pickle.load(f)

                return tracks
            else:
                print(f"[INFO] Stub file '{stub_path}' doesn't exists, creating one...")

        detections = self.detect_frames(frames)

        tracks = {
            "players" : [],
            "referees" : [],
            "ball" : []
        }

        for frame_num , detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "GoalKeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox" : bbox}

                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox" : bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {"bbox" : bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks , f)

        return tracks
    
    def team_draw_ball_controll(self ,frame , frame_num, team_ball_controll):
        overlay = frame.copy()
        cv.rectangle(overlay , (1350,850), (1900 , 970), (255,255,255) ,-1)
        cv.addWeighted(overlay, 0.4, frame , 0.6 , 0,frame)
        
        team_ball_controll_till_frame = team_ball_controll[:frame_num+1]

        team_1_num_frames = team_ball_controll_till_frame[team_ball_controll_till_frame==1].shape[0]
        team_2_num_frames = team_ball_controll_till_frame[team_ball_controll_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+ team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+ team_2_num_frames)

        cv.putText(frame , f"Team 1 Ball Controll: {team_1*100:.2f} %" , (1400, 900) , cv.FONT_HERSHEY_SIMPLEX , 1 , (0,0,0), 3)

        cv.putText(frame , f"Team 2 Ball Controll: {team_2*100:.2f} %" , (1400, 950) , cv.FONT_HERSHEY_SIMPLEX , 1 , (0,0,0), 3)    

        return frame    
        

    def draw_annotations(self, video_frames , tracks , team_ball_controll):
        output_video_frames = []

        for frame_num , frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict= tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            for track_id , player in player_dict.items():
                color = player.get('team_color', (0,0,255))
                frame = self.draw_ellipse(frame ,player["bbox"] , color, track_id)

                if player.get('has_ball' , False):
                    frame = self.draw_triangles(frame ,player["bbox"],(0,0,255))
            
            for _ , refree in referee_dict.items():
                frame = self.draw_ellipse(frame ,refree["bbox"] , (0,255,255))

            for track_id , ball in ball_dict.items():
                frame = self.draw_triangles(frame , ball['bbox'], (0,255,0))

            frame =  self.team_draw_ball_controll(frame , frame_num, team_ball_controll)
        
            output_video_frames.append(frame)

        return output_video_frames