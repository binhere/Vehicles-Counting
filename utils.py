import cv2
import numpy as np


def compute_center_of_zone(pts):
    '''
        Input:
            pts: [[x1, y1], [x2, y2], ..., [xn, yn]]    ;   xk, yk is int
    '''
    x_sum = 0
    y_sum = 0
    
    for point in pts:
        x_sum += point[0]
        y_sum += point[1]
        
    x_avg = int(x_sum / len(pts))
    y_avg = int(y_sum / len(pts))
    
    return (x_avg, y_avg)


def draw_zones(frame, id_color_zone_pairs, counter, border_thickness=2):
    alpha = 1
    overlay = frame.copy()
    
    for pair in id_color_zone_pairs:
        id = pair[0]
        r, g, b, alpha = pair[1]
        pts = pair[2]
        
        center_x, center_y = compute_center_of_zone(pts)
        
        pts = np.array(pts)
        cv2.polylines(frame, [pts], isClosed=True, color=(b, g, r), thickness=border_thickness)
        cv2.fillPoly(overlay, pts=[pts], color=(b, g, r))
        
    # Blend the original frame and the overlay with alpha transparency
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    for pair in id_color_zone_pairs:
        id = pair[0]
        r, g, b, alpha = pair[1]
        pts = pair[2]
            
        center_x, center_y = compute_center_of_zone(pts)
        cv2.putText(frame, f'Zone {id+1}: {counter[id]}', (center_x-10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        
    return frame


def load_zones(width, height, filename='config.txt'):
    id_color_zone_pairs = []
    
    with open(filename, 'r') as f:
        all_zones = f.read().split('\n')[:-1]
        
        # indexs of zones start from 0
        for id, zone in enumerate(all_zones):
            parts = zone.split(';')
            color, all_points = parts[0], parts[1:-1]
             
            r, g, b, alpha = color.split(',')
            r = int(r)
            g = int(g)
            b = int(b)
            alpha = float(alpha)
            
            pts = []
            for point in all_points:
                x, y = point.split(',')
                x = int(float(x) * width)
                y = int(float(y) * height)
                pts.append([x, y])
            
            id_color_zone_pairs.append([id, (r, g, b, alpha), pts])

    return id_color_zone_pairs


def calculate_distance(position1, position2):
    return ((position1[0] - position2[0]) ** 2 + (position1[1] - position2[1]) ** 2) ** 0.5


def update_heatmap(heatmap,x, y, w, h, id_box, track_history):
    current_position = (x, y)

    top_left_x = int(x - w / 2)
    top_left_y = int(y - h / 2)
    bottom_right_x = int(x + w / 2)
    bottom_right_y = int(y + h / 2)

    top_left_x = max(0, top_left_x)
    top_left_y = max(0, top_left_y)
    bottom_right_x = min(heatmap.shape[1], bottom_right_x)
    bottom_right_y = min(heatmap.shape[0], bottom_right_y)
   
    track_history[id_box].append(current_position)
    if len(track_history[id_box]) >= 2:
        last_position = track_history[id_box][-2]
        
        track_history[id_box].pop(0)
        
        if last_position and calculate_distance(last_position, current_position) >= 5:
            heatmap[top_left_y:bottom_right_y, top_left_x:bottom_right_x] += 1
    
    return heatmap, track_history

def save_heatmap(frame, heatmap, filename='heatmap.png', alpha=0.7):
    heatmap_blurred = cv2.GaussianBlur(heatmap, (15, 15), 0)
    heatmap_norm = cv2.normalize(heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_color, alpha, 0)
    cv2.imwrite(filename, overlay)
    
def save_result(counter):
    with open('result.txt', 'w') as f:
        for key, value in counter.items():
            f.write(f'Zone {key+1}: {value}' + "\n")