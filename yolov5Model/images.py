import cv2

# Initialize the VideoCapture object
rtsp_url = "rtsp://admin:iWizards@172.16.16.248/streaming/channels/101"
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

frame_count = 0
saved_image_count = 0

while True:
    ret, frame = cap.read()

    # Check if frame is read correctly
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    frame_count += 1

    # Save every 30th frame
    if frame_count % 30 == 0:
        saved_image_count += 1
        cv2.imwrite(f'saved_frame_{saved_image_count}.jpg', frame)
        print(f'Saved frame {saved_image_count}')

        # Stop after saving 10 images
        if saved_image_count == 10:
            break

# Release the VideoCapture object
cap.release()
