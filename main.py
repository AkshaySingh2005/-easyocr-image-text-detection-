import cv2
import easyocr
import matplotlib.pyplot as plt


img_path = './test1.jpg'


img = cv2.imread(img_path)


if img is None:
    raise FileNotFoundError(f"Image at path '{img_path}' could not be loaded.")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Detect text
detected_texts = reader.readtext(img)

threshold = 0.5

for detection in detected_texts:
    bbox, text, score = detection

    print(f"Detected text: {text}, Score: {score}, BBox: {bbox}")

    # Only process if confidence score is above the threshold
    if score > threshold:
        # Extract top-left and bottom-right from bounding box
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))

        # Draw bounding box
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

        # Position to insert text
        text_position = tuple(map(int, bbox[0]))
        cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

# Display the image with bounding boxes
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
