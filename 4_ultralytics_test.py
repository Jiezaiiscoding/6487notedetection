from collections import defaultdict

import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Dictionary to store tracking history with default empty lists
track_history = defaultdict(lambda: [])

# Load the YOLO model with segmentation capabilities
model = YOLO("note_0719.pt")
path = "video/test.mp4"
# Open the video file
cap = cv2.VideoCapture(path)

# Retrieve video properties: width, height, and frames per second
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
model.predict(source=path, save=True, imgsz=640, conf=0.5)

# Initialize video writer to save the output video with the specified properties
out = cv2.VideoWriter("instance-segmentation-object-tracking.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

while True:
    # Read a frame from the video
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break
#
#     # Create an annotator object to draw on the frame
#     annotator = Annotator(im0, line_width=2)
#
#     # Perform object tracking on the current frame
#     results = model.predict(im0, persist=True)
#     #
#     # # Check if tracking IDs and masks are present in the results
#     # if results[0].boxes.id is not None:
#     #     # Extract masks and tracking IDs
#     #     boxes = results[0].boxes.xywh
#     #     track_ids = results[0].boxes.id.int().cuda().tolist()
#     #
#     #     # Annotate each mask with its corresponding tracking ID and color
#     #     for box, track_id in zip(boxes, track_ids):
#     #         annotator.seg_bbox(mask=box,mask_color=colors(track_id,True),track_label=str(track_id))
#     #         # annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), track_label=str(track_id))
#
#     # Write the annotated frame to the output video
#     # out.write(im0)
#     # Display the annotated frame
#     cv2.imshow("object-tracking", im0)
#
#     # Exit the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#
# # Release the video writer and capture objects, and close all OpenCV windows
# # out.release()
# cap.release()
# cv2.destroyAllWindows()