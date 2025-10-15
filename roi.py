# In roi.py
import cv2
import numpy as np

def crop_leaf_roi(image_path, cache_path):
    """
    Reads an image, segments the leaf using HSV color masking and morphology,
    calculates an expanded bounding box, crops the leaf, resizes it to 224x224,
    and saves it to the cache path.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}. Skipping.")
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([20, int(0.15*255), int(0.15*255)])
    upper_green = np.array([95, 255, 255])
    
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    img_area = img.shape[0] * img.shape[1]
    mask_area = cv2.countNonZero(mask)
    
    if 0.05 * img_area < mask_area < 0.90 * img_area:
        x, y, w, h = cv2.boundingRect(mask)
        margin_x = int(w * 0.10)
        margin_y = int(h * 0.10)
        
        x_start = max(0, x - margin_x)
        y_start = max(0, y - margin_y)
        x_end = min(img.shape[1], x + w + margin_x)
        y_end = min(img.shape[0], y + h + margin_y)
        
        cropped_img = img[y_start:y_end, x_start:x_end]
    else:
        cropped_img = img

    resized_img = cv2.resize(cropped_img, (224, 224), interpolation=cv2.INTER_AREA)
    cv2.imwrite(cache_path, resized_img)
    return True
