import cv2 
import time
from tf_utils import detect_image
import streamlit as st
from PIL import Image

# Streamlit app
st.title("Yolov8 object detection")
cap = cv2.VideoCapture("rtsp://admin:Yahyoxonqwe1.@169.254.17.56:554/h264/ch1/main/av_stream")

def main():
    fps = 0
    prev_time = 0
    curr_time = 0
    for_fps = st.empty()
    outputing = st.empty()
    while True:
        ret, frame = cap.read()

        # Run the inference
        output = detect_image(frame)

        # Convert the output to an image that can be displayed
        output_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

        # Display the image
        outputing.image(output_image)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        for_fps.write(f"FPS: {fps}")
        # print(fps)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()