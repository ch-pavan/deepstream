import cv2
import json

# Initialize global variables
roi = []
drawing = False

def draw_line(event, x, y, flags, param):
    global line_points, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        line_points = [(x, y)]

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        line_points.append((x, y))
        cv2.line(img, line_points[0], line_points[1], (0, 255, 0), 2)
        cv2.imshow('Image', img)

# Load the image
img = cv2.imread('/home/ch/Downloads/deepstream-6.4/samples/streams/sample_720p.jpg')
cv2.imshow('Image', img)

# Set mouse callback function
cv2.setMouseCallback('Image', draw_line)

cv2.waitKey(0)
cv2.destroyAllWindows()
# Save the coordinates
# Scale factors
scale_x = 1920 / 1280
scale_y = 1080 / 720

scaled_line_coordinates = [
    [int(pt[0] * scale_x), int(pt[1] * scale_y)] for pt in line_points
]
line_dict = {"line": scaled_line_coordinates}

with open('roi_coordinates.json', 'w') as f:
    json.dump(line_dict, f)