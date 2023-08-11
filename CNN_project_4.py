import tensorflow as tf
from keras.preprocessing import image
import cv2
import numpy as np
from collections import deque


# Load the pre-trained face detection model (e.g., Haar Cascade classifier)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to load a new gender classification model
def load_gender_model(model_filename):
    from keras.models import load_model
    gender_model = load_model(model_filename)
    return gender_model

# Function to load a new age prediction model
def load_age_model(model_filename):
    # Load the model using appropriate method for your model type (e.g., Keras or TensorFlow)
    from keras.models import load_model
    age_model = load_model(model_filename)
    # age_model = ...
    return age_model
    
def load_emotion_model(model_filename):
    # Load the model using appropriate method for your model type (e.g., Keras or TensorFlow)
    from keras.models import load_model
    emotion_model = load_model(model_filename)
    return emotion_model
    
previous_predictions = deque(maxlen=100) 

# Function to make gender predictions using the loaded model
def predict_gender(face, model):
    face = cv2.resize(face, (64, 64))
    face_array = image.img_to_array(face)
    face_array = np.expand_dims(face_array, axis=0)
    gender_prediction = model.predict(face_array)
    return 'Male' if gender_prediction[0][0] > 0.5 else 'Female'

# Function to make age predictions using the loaded model
def predict_age(face, model):
    face = cv2.resize(face, (64, 64))
    face_array = image.img_to_array(face)
    face_array = np.expand_dims(face_array, axis=0)
    age_prediction = model.predict(face_array)
    age_index = np.argmax(age_prediction)  # Get the index of the predicted age
    # Map the index to the actual age value using your dataset's mapping
    # Replace this line with your dataset's mapping of age_index to age value
    age = str(age_index)  # For illustration purposes, age is converted to string
    
    age = str(age_index)  # For illustration purposes, age is converted to string

    # Add the current prediction to the deque
    previous_predictions.append(age_index)

    # Calculate the average prediction over the last several frames
    smoothed_age_index = int(np.mean(previous_predictions))

    # Map the smoothed age index to the actual age value using your dataset's mapping
    # Replace this line with your dataset's mapping of smoothed_age_index to age value
    smoothed_age = str(smoothed_age_index)  # For illustration purposes, age is converted to string

    return smoothed_age
    
def predict_emotion(face, model):
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_gray = cv2.resize(face_gray, (64, 64))  # Resize the cropped face region to 64x64 grayscale
    face_array = np.expand_dims(face_gray, axis=0)
    # Stack the single-channel image three times to create an RGB image
    face_array_rgb = np.stack([face_array] * 3, axis=-1)
    face_array_rgb = face_array_rgb.astype('float32') / 255.0  # Normalize the image data
    emotion_prediction = model.predict(face_array_rgb)
    emotion_index = np.argmax(emotion_prediction)  # Get the index of the predicted emotion

    # Map the index to the actual emotion label using your dataset's mapping
    # Replace this line with your dataset's mapping of emotion_index to emotion label
    emotions = ['Angry', 'Happy', 'Neutral','Sad']
    predicted_emotion = str(emotions[emotion_index])

    return predicted_emotion

# Open the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera (you can change it to another camera index if available)

# Load the gender classification model
gender_model = load_gender_model('my_cnn_model.h5')

# Load the age prediction model
age_model = load_age_model('my_cnn_model2.h5')

emotion_model = load_emotion_model('my_cnn_model3.h5')

# Create a small window to display the camera feed
cv2.namedWindow('Gender,Age and Mood Prediction', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Gender,Age and Mood Prediction', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

# Flag to keep track of window state
is_fullscreen = False

def on_maximize(x, y):
    global is_fullscreen
    if not is_fullscreen:
        cv2.setWindowProperty('Gender,Age and Mood Prediction', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        is_fullscreen = True
    else:
        cv2.setWindowProperty('Gender,Age and Mood Prediction', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        is_fullscreen = False

# Set callback for mouse events
cv2.setMouseCallback('Gender,Age and Mood Prediction', on_maximize)

cv2.namedWindow('Gender, Age, and Emotion Prediction', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Gender, Age, and Emotion Prediction', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)


while True:
    # Capture frame-by-frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame using the face detection model
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Crop the face region
        face_roi = frame[y:y+h, x:x+w]

        # Make gender prediction for the face using the loaded gender model
        gender = predict_gender(face_roi, gender_model)

        # Make age prediction for the face using the loaded age model
        age = predict_age(face_roi, age_model)

        # Make emotion prediction for the face using the loaded emotion model
        emotion = predict_emotion(face_roi, emotion_model)

        # Display the gender, age, and emotion predictions on the frame
        cv2.putText(frame, gender, (x, y-70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, age, (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the live camera feed with gender, age, and emotion predictions
    cv2.imshow('Gender, Age, and Emotion Prediction', frame)


    # Check for a keyboard event and exit if 'q' key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty('Gender, Age, and Emotion Prediction', cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
