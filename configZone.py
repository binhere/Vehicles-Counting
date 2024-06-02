import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas


################### Functions ###################

def hex_to_rgba(hex_color, alpha=0.5):   
    hex_color = hex_color.lstrip('#')
    
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    
    rgba = f'rgba({red}, {green}, {blue}, {alpha})'
    return rgba


def extract_color_pos(df, filename="config.txt"):   
    with open(filename, 'w') as f:  
        configuration = ""
        
        for i, row in df.iterrows():
            r, g, b, alpha = row['fill'].strip('rgba()').split(', ')
            raw_pos = row['path'].strip('[]').split('], [')
            
            handled_pos = ""
            for i, string in enumerate(raw_pos):
                parts = string.split(', ')
                if parts[0] != "'z'":
                    handled_pos += f'{float(parts[1]) / MAX_WIDTH},{float(parts[2]) / MAX_HEIGHT};' 
                
            configuration += f'{r},{g},{b},{alpha}' + ";" + handled_pos + "\n"
            
        f.write(configuration)  
        
        
        
################### Main ###################

MAX_HEIGHT = 400
MAX_WIDTH = 600

new_height = None
new_width = None

img_width = None
img_height = None


drawing_mode = 'polygon'
stroke_width = st.sidebar.slider("Stroke width: ", 0, 5, 1)
if drawing_mode == "point":
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
color = st.sidebar.color_picker("Color", "#EEEEEE")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
# realtime_update = st.sidebar.checkbox("Update in realtime", True)
canvas_result = None


# Create a canvas component
if bg_image is not None:
    uploaded_img = Image.open(bg_image)
    img_width, img_height = uploaded_img.size
    
    canvas_result = st_canvas(
        fill_color=hex_to_rgba(color),
        stroke_width=stroke_width,
        stroke_color=color,
        background_image=uploaded_img,
        # update_streamlit=realtime_update,
        update_streamlit=True,
        height = MAX_HEIGHT,
        width = MAX_WIDTH,
        drawing_mode="polygon",
        point_display_radius=point_display_radius if drawing_mode == "point" else 0,
        # display_toolbar=st.sidebar.checkbox("Display toolbar", True),
        display_toolbar=True,
        key="full_app",
    )
else:
    html_instruction = """
        <h1 style='text-align: center; font-size: 20px; padding-top: 40px;'>
            Please upload a background image to draw on it.
        </h1>
    """
    st.markdown(html_instruction, unsafe_allow_html=True)

# Do something interesting with the image data and paths
if canvas_result is not None and canvas_result.json_data is not None:
    objects = pd.json_normalize(canvas_result.json_data["objects"])
    for col in objects.select_dtypes(include=["object"]).columns:
        objects[col] = objects[col].astype("str")
    
    if len(objects.columns.to_list()):
        objects = objects.loc[:, ['fill', 'path']]
        
        extract_color_pos(objects)
        st.dataframe(objects)

 