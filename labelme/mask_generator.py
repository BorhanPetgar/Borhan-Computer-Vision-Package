import numpy as np
import cv2
import json

# Sample JSON data
json_path = '/home/borhan/Documents/code_workspace/vm6_backup/projects/anomaly/repos/DCAMA/good (copy)/other3/358425454064707_G1_H5/back.json'
with open(json_path, 'r') as file:
    json_data = file.read()
# Load JSON data
data = json.loads(json_data)

# Determine the size of the mask (you can adjust this based on your needs)
# mask_height = int(max([max(point[1] for point in shape['points']) for shape in data['shapes']])) + 10
# mask_width = int(max([max(point[0] for point in shape['points']) for shape in data['shapes']])) + 10
mask_height = data['imageHeight']
mask_width = data['imageWidth']
# Create an empty mask
mask = np.zeros((mask_height, mask_width), dtype=np.uint8)

# Draw each polygon on the mask
for shape in data['shapes']:
    points = np.array(shape['points'], dtype=np.int32)
    cv2.fillPoly(mask, [points], color=255)

# Save or display the mask
cv2.imwrite('mask_back.png', mask)
# cv2.imshow('Binary Mask', mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
