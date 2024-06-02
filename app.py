import cv2
import streamlit as st
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from utils import *
import os


#################### SET UP ####################

global TRACKER_BYTETRACK, TRACKER_BOTSORT, PATH_TRACKER, TRACKER, PATH_MODEL, MODEL_NAME, PATH_VID

PATH_TRACKER = 'tracker'
PATH_BYTETRACK = os.path.join(PATH_TRACKER, 'bytetrack.yaml')
PATH_BOTSORT =  os.path.join(PATH_TRACKER, 'botsort.yaml') 
TRACKER = PATH_BYTETRACK # choose tracker here

MODEL_NAME = 'hcmdata_8n_50e.pt'   # choose model here
PATH_MODEL = os.path.join('model', MODEL_NAME)

PATH_VID = "cam_01.mp4" # choose video for inference here

#################### MAIN ####################


cap = cv2.VideoCapture(PATH_VID)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

model = YOLO(PATH_MODEL)
id_color_zone_pairs = load_zones(frame_width, frame_height)

frame_placeholder = st.empty()

id_counted = []
track_history = defaultdict(lambda: [])
counter = {i: 0 for i in range(len(id_color_zone_pairs))}

heatmap = np.zeros((frame_height, frame_width, 3), dtype=np.float32)

video_output = cv2.VideoWriter('output/output_video.mp4', fourcc, fps, (frame_width, frame_height))

selected_model = st.sidebar.selectbox("model", 
                                      ("YOLOv8n custom",))

selected_tracker = st.sidebar.selectbox("tracker", 
                                        ("bytetrack", "botsort"))

TRACKER = PATH_BYTETRACK if selected_tracker == "bytetrack" else PATH_BOTSORT

conf_score = st.sidebar.slider(label='confidence', 
                               min_value=0.0, 
                               max_value=1.0)

clicked_button = st.sidebar.button('stop video')

while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret:  
        results = model.track(frame, persist=True, verbose=False, tracker=TRACKER, conf=conf_score)
        annotated_frame = results[0].plot()
        
        if results[0].boxes.is_track == False:
            continue
        
        id_in_frame = []
        for box in results[0].boxes.cpu().numpy():
            x, y, w, h = box.xywh.astype(int)[0]
            id_box = box.id.astype(int)[0]
            id_cls = box.cls.astype(int)[0]
            id_in_frame.append(id_box)
            
            heatmap, track_history = update_heatmap(heatmap, x, y, w, h, id_box, track_history)
            
            for pair in id_color_zone_pairs:
                id_zone = pair[0]
                zone = pair[2]
                
                pts = np.array(zone, np.int32)
                # pts = pts.reshape((-1,1,2))
                center_point = (float(x), float(y))
                dist = cv2.pointPolygonTest(pts, center_point, False)

                if dist > 0:
                    if id_box not in id_counted:
                        id_counted.append(id_box)
                        counter[id_zone] += 1                        
                        break
                    
                if id_box not in id_counted:
                    annotated_frame = cv2.circle(annotated_frame, (x, y), radius=2, color=(0, 255, 255), thickness=6)
                else:
                    annotated_frame = cv2.circle(annotated_frame, (x, y), radius=2, color=(0, 255, 0), thickness=6)
                    
        
        annotated_frame = draw_zones(annotated_frame, id_color_zone_pairs, counter)
       
        fps = 1 / (sum(results[0].speed.values())/1000)
        fps_text = f'FPS: {fps:.2f}'
        cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        total_vehicles = [val for key, val in counter.items()]
        cv2.putText(annotated_frame, f'Total: {sum(total_vehicles)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # video_output.write(annotated_frame)
        # cv2.imshow('output per frame', annotated_frame)
        frame_placeholder.image(annotated_frame, channels="BGR")
        
        if cv2.waitKey(1) == ord('q') or clicked_button:
            video_output.release()
            
            save_heatmap(frame, heatmap)
            print('>> saved heatmap')
            
            save_result(counter)
            print('>> saved result')
            
            print('>> exiting')
            break
        
    else:
        save_heatmap(frame, heatmap)
        print('>> saved heatmap')
        
        save_result(counter)
        print('>> saved result')
        
        video_output.release()
        print('>> video ended')
        break
        
        
cap.release()
cv2.destroyAllWindows()