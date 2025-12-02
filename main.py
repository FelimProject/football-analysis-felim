from utils.video_utils import read_video, save_video
from trackers.tracker import Tracker
from team_assinger.team_assigner import TeamAssinger
from player_ball_assigner.player_ball_assinger import PlayerBallAssinger
from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from view_transformer.view_transformer import ViewTransformer
from speed_and_distance_estimator.speed_and_distance_estimator import SpeedAndDistanceEstimator
import cv2 as cv
import numpy as np

def main() :
    video_frames = read_video("input_video/08fd33_4.mp4")

    tracker = Tracker("models/best.pt")

    tracks = tracker.get_object_tracks(
        video_frames, 
        True, 
        'stubs/track_stubs.pkl'
    )

    tracker.add_position_to_tracks(tracks)

    camera_movement_estimator = CameraMovementEstimator(video_frames[0])


    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/camera_movement_stub.pkl'
    )

    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    view_transformer =  ViewTransformer()
    view_transformer.add_transformer_position_to_tracks(tracks)

    tracks['ball'] = tracker.interploate_ball_position(tracks['ball'])

    speed_and_distance_estimator = SpeedAndDistanceEstimator()

    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    team_assigner = TeamAssinger()
    team_assigner.assing_team_color(video_frames[0] , tracks['players'][0])

    for frame_num , player_track in enumerate(tracks['players']):
        for player_id , track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], 
                track['bbox'], 
                player_id
            )

            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    
    player_assigner = PlayerBallAssinger()
    team_ball_controll = []
    for frame_num , player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']

        assigner_player_id = player_assigner.assing_ball_to_player(player_track, ball_bbox)
        
        if assigner_player_id != -1 :
            tracks['players'][frame_num][assigner_player_id]['has_ball'] = True 
            team_ball_controll.append(tracks['players'][frame_num][assigner_player_id]['team'])
        else :
            team_ball_controll.append(team_ball_controll[-1])

    team_ball_controll = np.array(team_ball_controll)


    output_video_frames = tracker.draw_annotations(
        video_frames, 
        tracks, 
        team_ball_controll
    )

    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, 
        camera_movement_per_frame
    )

    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    save_video(output_video_frames , 'output_video/output_video.avi')

if __name__ == "__main__":
    main()
