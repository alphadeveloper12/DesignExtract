from transformers import pipeline
from PIL import Image
import numpy as np
import os
import cv2
import cv2 as cv
from django.core.files.storage import FileSystemStorage

from ultralytics import YOLO
import time
from django.conf import settings
import torch
from PIL import Image
from skimage import io
from huggingface_hub import hf_hub_download
from .briarmbg import BriaRMBG
from .new import preprocess_image, postprocess_image
import os
import uuid
import os
import cv2 as cv
import numpy as np
from PIL import Image
import torch
import uuid
from .briarmbg import BriaRMBG
from huggingface_hub import hf_hub_download

def process_image(file_path):
    shapes_output_dir = 'media/shapes'
    class_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush', 'orange', 'flower', 'petal-petal', 'petal', 'leaves'
    ]

    true_positives = {name: 0 for name in class_names}
    false_positives = {name: 0 for name in class_names}
    false_negatives = {name: 0 for name in class_names}

    start_time = time.time()

    # Load the image
    image = Image.open(file_path)

    # Initialize the image-to-text pipeline
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    # Generate a caption for the image
    caption = captioner(image, max_new_tokens=50)
    print("Caption:", caption)

    # Load YOLOv5 model
    model = YOLO('yolov5su.pt')

    # Perform object detection
    results = model(file_path)

    # Create directory for shapes if it doesn't exist
    os.makedirs(shapes_output_dir, exist_ok=True)

    # Delete all existing files in the 'shapes' folder
    for existing_image in os.listdir(shapes_output_dir):
        os.remove(os.path.join(shapes_output_dir, existing_image))

    detected_classes = []
    if results:
        detections = results[0].boxes.data.cpu().numpy()
        for i, (x1, y1, x2, y2, conf, cls) in enumerate(detections):
            if conf > 0.5:
                cls_name = class_names[int(cls)]
                detected_classes.append(cls_name)
                cropped_img = image.crop((x1, y1, x2, y2))
                cropped_img.save(f'{shapes_output_dir}/{cls_name}_{i}.jpeg')

        detected_classes_set = set(detected_classes)

        for name in detected_classes_set:
            true_positives[name] = 1
            false_positives[name] = 0
            false_negatives[name] = 0

        for name in class_names:
            if name not in detected_classes_set:
                false_negatives[name] = 1

        precision = {name: true_positives[name] / (true_positives[name] + false_positives[name]) if (true_positives[
                                                                                                         name] +
                                                                                                     false_positives[
                                                                                                         name]) > 0 else 0
                     for name in detected_classes_set}
        recall = {name: true_positives[name] / (true_positives[name] + false_negatives[name]) if (true_positives[name] +
                                                                                                  false_negatives[
                                                                                                      name]) > 0 else 0
                  for name in detected_classes_set}
        f1 = {name: 2 * (precision[name] * recall[name]) / (precision[name] + recall[name]) if (precision[name] +
                                                                                                recall[name]) > 0 else 0
              for name in detected_classes_set}

    img = cv2.imread(file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 500
    for i, contour in enumerate(contours):
        if i == 0:
            continue

        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        contour_area = cv2.contourArea(contour)
        if contour_area > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            shape_img = np.zeros((h, w), dtype=np.uint8)
            contour_offset = contour - contour.min(axis=0)
            cv2.drawContours(shape_img, [contour_offset], 0, (255), 1)
            shape_filename = os.path.join(shapes_output_dir, f'shape_{i}.png')
            cv2.imwrite(shape_filename, shape_img)

    annotated_image_path = os.path.join(shapes_output_dir, 'annotated_image.jpg')
    cv2.imwrite(annotated_image_path, img)

    end_time = time.time()
    computational_time = end_time - start_time

    return {
        'caption': caption,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'speed': computational_time,
        'processed_images': [os.path.join('shapes', f) for f in os.listdir(shapes_output_dir)]
    }


def process_image1(file_path):
    shapes_output_dir = 'media/shapes1'
    class_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'
    ]

    true_positives = {name: 0 for name in class_names}
    false_positives = {name: 0 for name in class_names}
    false_negatives = {name: 0 for name in class_names}

    start_time = time.time()

    # Load the image
    image = Image.open(file_path)

    # Initialize the image-to-text pipeline
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    # Generate a caption for the image
    caption = captioner(image, max_new_tokens=50)
    print("Caption:", caption)

    # Load YOLOv5 model
    model = YOLO('yolov5su.pt')

    # Perform object detection
    results = model(file_path)

    # Create directory for shapes if it doesn't exist
    os.makedirs(shapes_output_dir, exist_ok=True)

    # Delete all existing files in the 'shapes1' folder
    for existing_image in os.listdir(shapes_output_dir):
        os.remove(os.path.join(shapes_output_dir, existing_image))

    detected_classes = []
    if results:
        detections = results[0].boxes.data.cpu().numpy()
        for i, (x1, y1, x2, y2, conf, cls) in enumerate(detections):
            if conf > 0.5:
                cls_name = class_names[int(cls)]
                detected_classes.append(cls_name)
                cropped_img = image.crop((x1, y1, x2, y2))
                # Convert image to RGB mode if necessary
                if cropped_img.mode != 'RGB':
                    cropped_img = cropped_img.convert('RGB')
                cropped_img.save(f'{shapes_output_dir}/{cls_name}_{i}.jpeg')

        detected_classes_set = set(detected_classes)

        for name in detected_classes_set:
            true_positives[name] = 1
            false_positives[name] = 0
            false_negatives[name] = 0

        for name in class_names:
            if name not in detected_classes_set:
                false_negatives[name] = 1

        precision = {name: true_positives[name] / (true_positives[name] + false_positives[name]) if (true_positives[
                                                                                                         name] +
                                                                                                     false_positives[
                                                                                                         name]) > 0 else 0
                     for name in detected_classes_set}
        recall = {name: true_positives[name] / (true_positives[name] + false_negatives[name]) if (true_positives[name] +
                                                                                                  false_negatives[
                                                                                                      name]) > 0 else 0
                  for name in detected_classes_set}
        f1 = {name: 2 * (precision[name] * recall[name]) / (precision[name] + recall[name]) if (precision[name] +
                                                                                                recall[name]) > 0 else 0
              for name in detected_classes_set}

    img = cv2.imread(file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 500
    for i, contour in enumerate(contours):
        if i == 0:
            continue

        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        contour_area = cv2.contourArea(contour)
        if contour_area > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            shape_img = np.zeros((h, w), dtype=np.uint8)
            contour_offset = contour - contour.min(axis=0)
            cv2.drawContours(shape_img, [contour_offset], 0, (255), 1)
            shape_filename = os.path.join(shapes_output_dir, f'shape_{i}.png')
            cv2.imwrite(shape_filename, shape_img)

    annotated_image_path = os.path.join(shapes_output_dir, 'annotated_image.jpg')
    cv2.imwrite(annotated_image_path, img)

    end_time = time.time()
    computational_time = end_time - start_time

    return {
        'caption': caption,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'speed': computational_time,
        'processed_images': [os.path.join('shapes1', f) for f in os.listdir(shapes_output_dir)]
    }


def remove_background(image_path):
    # Load and prepare the model
    model_path = hf_hub_download("briaai/RMBG-1.4", 'model.pth')

    net = BriaRMBG()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    # Prepare input
    model_input_size = [1024, 1024]
    orig_im = Image.open(image_path).convert("RGB")
    orig_im = np.array(orig_im)
    orig_im_size = orig_im.shape[0:2]
    processed_image = preprocess_image(orig_im, model_input_size).to(device)

    # Inference
    with torch.no_grad():
        result = net(processed_image)

    # Post-process
    result_image = postprocess_image(result[0][0], orig_im_size)

    # Ensure result_image is a binary mask: 0 for background, 255 for object
    result_image = (result_image > 0).astype(np.uint8) * 255

    # Convert the original image to RGBA (with alpha channel)
    orig_image_rgba = Image.open(image_path).convert("RGBA")

    # Create a transparent image
    no_bg_image = Image.new("RGBA", orig_image_rgba.size, (0, 0, 0, 0))

    # Use the binary mask to paste the original image onto the transparent background
    mask = Image.fromarray(result_image).convert("L")  # 'L' mode is for grayscale (binary mask)

    # Paste the original image onto the transparent image using the binary mask
    no_bg_image.paste(orig_image_rgba, (0, 0), mask)

    # Save the result with a transparent background
    output_filename = f"{uuid.uuid4()}.png"
    response_image_path = os.path.join('/tmp/', output_filename)
    no_bg_image.save(response_image_path)

    return response_image_path


def process_image_additional(image_path):
    output_folder = os.path.join(settings.MEDIA_ROOT, 'shapes')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    img = cv.imread(image_path)
    if img is None:
        return {'processed_images': [], 'error': 'Could not read image'}

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.blur(gray, (10, 10))
    ret, thresh = cv.threshold(blur, 1, 255, cv.THRESH_OTSU + cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    idx = 0
    area_threshold = 2500
    processed_images = []

    # Temporary directory for cropped images
    temp_cropped_folder = os.path.join(output_folder, 'temp_cropped')
    if not os.path.exists(temp_cropped_folder):
        os.makedirs(temp_cropped_folder)

    for c in contours:
        area = cv.contourArea(c)
        if area > area_threshold:
            x, y, w, h = cv.boundingRect(c)
            cropped_img = img[y:y + h, x:x + w]
            temp_cropped_file_name = os.path.join(temp_cropped_folder, f'shape_{idx:03d}.png')
            cv.imwrite(temp_cropped_file_name, cropped_img)

            # Remove background from the cropped image
            image_path_no_bg = remove_background(temp_cropped_file_name)

            # Save the result after background removal
            processed_image = cv.imread(image_path_no_bg)
            if processed_image is not None:
                output_file_name = os.path.join(output_folder, f'shape_{idx:03d}.png')
                cv.imwrite(output_file_name, processed_image)
                processed_images.append('/media/' + os.path.relpath(output_file_name, settings.MEDIA_ROOT))
                idx += 1

    # Draw contours on the original image
    contoured_image_path = os.path.join(output_folder, 'contoured_image.png')
    cv.drawContours(img, contours, -1, (0, 255, 0), 5)
    cv.imwrite(contoured_image_path, img)

    # Append the relative path of the contoured image with '/media/' prefix
    processed_images.append('/media/' + os.path.relpath(contoured_image_path, settings.MEDIA_ROOT))
    contoured_image_url = '/media/' + os.path.relpath(contoured_image_path, settings.MEDIA_ROOT)

    # Clean up temporary directory
    for file_name in os.listdir(temp_cropped_folder):
        file_path = os.path.join(temp_cropped_folder, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

    os.rmdir(temp_cropped_folder)  # Remove the temporary directory itself if it's empty

    return {'processed_images': processed_images, 'contoured_image_path': contoured_image_url}
