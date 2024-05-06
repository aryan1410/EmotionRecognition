import cv2
import numpy as np
from keras.models import model_from_json
import pygame
import threading
pygame.mixer.init()
happy_song = 'expressions/happy.mp3'
disgusted_song = 'expressions/disgusted.mp3'
fear_song = 'expressions/fear.mp3'
angry_song = 'expressions/angry.mp3'
sad_song = 'expressions/sad.mp3'
surprised_song = 'expressions/surprised.mp3'

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('model/emotion_model_.json', 'r') ##Insert model.json path here
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5") ##Insert model weight path here
print("Loaded model from disk")

# start the webcam feed
#cap = cv2.VideoCapture(0)

# pass here your video path
# you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/

happy_song_playing = False
angry_song_playing = False
sad_song_playing = False
surprised_song_playing = False
disgusted_song_playing = False
fear_song_playing = False

def play_happy_song():
    global happy_song_playing
    # Play happy song
    pygame.mixer.music.load(happy_song)
    pygame.mixer.music.play()
    happy_song_playing = True

def play_angry_song():
    global angry_song_playing
    # Play happy song
    pygame.mixer.music.load(angry_song)
    pygame.mixer.music.play()
    angry_song_playing = True

def play_sad_song():
    global sad_song_playing
    # Play happy song
    pygame.mixer.music.load(sad_song)
    pygame.mixer.music.play()
    sad_song_playing = True

def play_surprised_song():
    global surprised_song_playing
    # Play happy song
    pygame.mixer.music.load(surprised_song)
    pygame.mixer.music.play()
    surprised_song_playing = True

def play_disgusted_song():
    global disgusted_song_playing
    # Play happy song
    pygame.mixer.music.load(disgusted_song)
    pygame.mixer.music.play()
    disgusted_song_playing = True

def play_fear_song():
    global fear_song_playing
    # Play happy song
    pygame.mixer.music.load(fear_song)
    pygame.mixer.music.play()
    fear_song_playing = True
    
cap = cv2.VideoCapture(0)

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(rgb_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_rgb_frame = rgb_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_rgb_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        if emotion_dict[maxindex] == 'Angry':
            # Check if happy song is already playing
            if not angry_song_playing:
                # Start playing happy song in a separate thread
                threading.Thread(target=play_angry_song).start()
        else:
            # Stop happy song if it's currently playing
            if angry_song_playing:
                pygame.mixer.music.stop()
                angry_song_playing = False
                
        if emotion_dict[maxindex] == 'Disgusted':
            # Check if happy song is already playing
            if not disgusted_song_playing:
                # Start playing happy song in a separate thread
                threading.Thread(target=play_disgusted_song).start()
        else:
            # Stop happy song if it's currently playing
            if disgusted_song_playing:
                pygame.mixer.music.stop()
                disgusted_song_playing = False
                
        if emotion_dict[maxindex] == 'Fearful':
            # Check if happy song is already playing
            if not fear_song_playing:
                # Start playing happy song in a separate thread
                threading.Thread(target=play_fear_song).start()
        else:
            # Stop happy song if it's currently playing
            if fear_song_playing:
                pygame.mixer.music.stop()
                fear_song_playing = False
                
        if emotion_dict[maxindex] == 'Happy':
            # Check if happy song is already playing
            if not happy_song_playing:
                # Start playing happy song in a separate thread
                threading.Thread(target=play_happy_song).start()
        else:
            # Stop happy song if it's currently playing
            if happy_song_playing:
                pygame.mixer.music.stop()
                happy_song_playing = False
                
        if emotion_dict[maxindex] == 'Sad':
            # Check if happy song is already playing
            if not sad_song_playing:
                # Start playing happy song in a separate thread
                threading.Thread(target=play_sad_song).start()
        else:
            # Stop happy song if it's currently playing
            if sad_song_playing:
                pygame.mixer.music.stop()
                sad_song_playing = False
                
        if emotion_dict[maxindex] == 'Surprised':
            # Check if happy song is already playing
            if not surprised_song_playing:
                # Start playing happy song in a separate thread
                threading.Thread(target=play_surprised_song).start()
        else:
            # Stop happy song if it's currently playing
            if surprised_song_playing:
                pygame.mixer.music.stop()
                surprised_song_playing = False
        
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()