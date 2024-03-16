import cv2
vcap = cv2.VideoCapture("rtsp://admin:iWizards@172.16.16.248/streaming/channels/101", cv2.CAP_FFMPEG)
while(1):
    ret, frame = vcap.read()
    if ret == False:
        print("Frame is empty")
        break
    else:
        cv2.imshow('VIDEO', frame)
        cv2.waitKey(1)