import os
import joblib
import json
import numpy as np
import cv2
# from . import wavelet
import pywt

__class_name_to_number = {}
__class_number_to_name = {}

__model = None

def w2d(img, mode='haar', level=1):
    imArray = img
    # Datatype conversions
    # Convert to grayscale
    if len(imArray.shape) > 2: # Check if image is not already grayscale
        imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    # Convert to float
    imArray = np.float32(imArray)
    imArray /= 255
    # Compute coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    # Process Coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    # Reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H

from sklearn.preprocessing import StandardScaler

def classify_image(file_path):
    img = get_cropped_image_if_2_eyes(file_path)
    result = []
    scalled_raw_img = cv2.resize(img, (32, 32))
    # print(scalled_raw_img)
    img_har = w2d(img, 'db1', 5)
    scalled_img_har = cv2.resize(img_har, (32, 32))
    
    # Flatten the images
    scalled_raw_img_flat = scalled_raw_img.flatten()
    scalled_img_har_flat = scalled_img_har.flatten()
    
    # Concatenate the flattened images
    combined_img = np.concatenate((scalled_raw_img_flat, scalled_img_har_flat))
    

    final = combined_img.reshape(1,-1).astype(float)
    
    # print("Final Shape:", final.shape)
    # print("Final Data:", final)

    prediction = __model.predict(final)
    print("Prediction:", prediction)

    class_probability = np.around(__model.predict_proba(final) * 100, 2).tolist()[0]
    print("Class Probabilities:", class_probability)

    class_name = class_number_to_name(prediction[0])
    print("Class Name:", class_name)

    result.append({
        'class': class_name,
        'class_probability': class_probability,
        'class_dictionary': __class_name_to_number
    })

    return result[0]['class_probability']







def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def load_saved_artifacts():
    current_dir = os.path.dirname(__file__)
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name
    global __model

    class_dict_file = os.path.join(current_dir, "class_dictionary.json")
    with open(class_dict_file, "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    # Load model
    if __model is None:
        model_file = os.path.join(current_dir, 'saved_model.pkl')
        with open(model_file, 'rb') as f:
            __model = joblib.load(f)
    print("loading saved artifacts...done")

def get_cropped_image_if_2_eyes(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        roi_color = img[y:y+h, x:x+w]
        return roi_color