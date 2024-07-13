import os
import cv2
import threading
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
from flask_cors import CORS
from flask import Flask, Response, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import moviepy.editor as me
import speech_recognition as sr
import librosa
import numpy as np
import contextlib
import wave

app = Flask(__name__)
log_file_path = "log.txt"
vocal_log_file_path = "logs2.txt"
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
mp_holistic = mp.solutions.holistic  # Mediapipe Solutions

with open('RandomForestClassifie.pkl', 'rb') as f:
    model = pickle.load(f)

detected_classes = []  # List to store detected class names

# Class names for the first set
class_names_set1 = ['Open body language', 'Neutral body language', 'Closed body language']
# Class names for the second set
class_names_set2 = ['Smile', 'Eye contact', 'Space occupation']

# Function to calculate probabilities
def calculate_probabilities(detected_classes, class_names):
    counts = {class_name: detected_classes.count(class_name) for class_name in class_names}
    total_counts = sum(counts.values())
    probabilities = {class_name: counts[class_name] / total_counts if total_counts > 0 else 0 for class_name in class_names}
    return probabilities

def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make Detections
            results = holistic.process(image)

            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                      mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))

            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            # Export coordinates
            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                # Extract Face landmarks
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                # Concatenate rows
                row = pose_row + face_row

                # Make Detections
                X = pd.DataFrame([row])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]

                # Store the detected class name in the list
                detected_classes.append(body_language_class)

                # Calculate probabilities for both sets of classes
                probabilities_set1 = calculate_probabilities(detected_classes, class_names_set1)
                probabilities_set2 = calculate_probabilities(detected_classes, class_names_set2)

                log_data = f"Class: {body_language_class}, Probabilities Set 1: {probabilities_set1}, Probabilities Set 2: {probabilities_set2}"
                log_message(log_data)

                # Grab ear coords
                coords = tuple(np.multiply(
                    np.array(
                        (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                         results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                    , [640, 480]).astype(int))

                cv2.rectangle(image,
                              (coords[0], coords[1] + 5),
                              (coords[0] + len(body_language_class) * 20, coords[1] - 30),
                              (245, 117, 16), -1)
                cv2.putText(image, body_language_class, coords,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Get status box
                cv2.rectangle(image, (0, 0), (300, 150), (245, 117, 16), -1)

                # Display Class
                cv2.putText(image, 'CLASS'
                            , (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, body_language_class.split(' ')[0]
                            , (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Display Probability
                cv2.putText(image, 'PROB'
                            , (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2))
                            , (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Display probabilities of specific classes for the first set
                for i, class_name in enumerate(class_names_set1):
                    cv2.putText(image, f'{class_name}: {probabilities_set1[class_name]:.2f}', (10, 60 + (i * 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # Display probabilities of specific classes for the second set
                for i, class_name in enumerate(class_names_set2):
                    cv2.putText(image, f'{class_name}: {probabilities_set2[class_name]:.2f}', (10, 120 + (i * 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            except Exception as e:
                print("Exception:", e)
                pass

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def extract_audio(video_file, audio_file):
    video = me.VideoFileClip(video_file)
    video.audio.write_audiofile(audio_file)

# Function to recognize speech from audio in French
def recognize_speech(audio_file, language='fr-FR'):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio, language=language)
        return text
    except sr.UnknownValueError:
        return "La reconnaissance vocale n'a pas pu comprendre l'audio"
    except sr.RequestError as e:
        return f"Impossible de demander les résultats du service de reconnaissance vocale Google; {e}"

# Function to analyze vocal qualities using librosa
def analyze_vocal_quality(audio_file):
    y, sr_librosa = librosa.load(audio_file)
    pitch, magnitude = librosa.core.piptrack(y=y, sr=sr_librosa)
    pitches = pitch[pitch > 0]
    mean_pitch = np.mean(pitches)
    rms = librosa.feature.rms(y=y)
    mean_intensity = np.mean(rms) * 100  # Scaling to dB
    
    # Vocal modulation analysis
    modulation = np.std(pitches)
    
    return mean_pitch / 7, mean_intensity * 200, modulation / 7

# Function to evaluate pace and pauses
def analyze_pace_and_pauses(audio_file, transcript):
    with contextlib.closing(wave.open(audio_file, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    words = transcript.split()
    num_words = len(words)
    pace = num_words / duration  # words per second

    # Pauses detection using librosa
    y, sr_librosa = librosa.load(audio_file)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr_librosa)
    times = librosa.times_like(onset_env, sr=sr_librosa)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr_librosa)
    pause_times = np.diff(times[beats])
    
    return pace, pause_times

# Function to analyze mean pitch
def analyze_mean_pitch(mean_pitch, gender='male'):
    if gender == 'male':
        if mean_pitch > 150:
            return "Hauteur moyenne élevée (indiquant de la nervosité ou de l'excitation)"
        elif 100 <= mean_pitch <= 150:
            return "Hauteur moyenne modérée (indiquant du calme et de la confiance)"
        else:
            return "Hauteur moyenne basse (indiquant de l'autorité et de la relaxation)"
    elif gender == 'female':
        if mean_pitch > 225:
            return "Hauteur moyenne élevée (indiquant de la nervosité ou de l'excitation)"
        elif 165 <= mean_pitch <= 225:
            return "Hauteur moyenne modérée (indiquant du calme et de la confiance)"
        else:
            return "Hauteur moyenne basse (indiquant de l'autorité et de la relaxation)"
    else:
        return "Genre inconnu pour l'analyse"

# Function to analyze mean intensity
def analyze_mean_intensity(mean_intensity):
    if mean_intensity < 60:
        return "Intensité moyenne basse (indiquant du calme, de l'introspection, de la douceur ou de la nervosité)"
    elif 60 <= mean_intensity <= 75:
        return "Intensité moyenne modérée (courante pour la conversation quotidienne, les conférences, les présentations)"
    else:
        return "Intensité moyenne élevée (indiquant de l'enthousiasme, de la confiance, de l'affirmation ou de l'urgence)"

# Function to analyze pitch variability (modulation)
def analyze_modulation(modulation, gender='male'):
    if modulation < 20:
        return "Faible variabilité (suggère une livraison monotone, calme ou sérieuse)"
    elif 20 <= modulation <= 40:
        return "Variabilité modérée (convoie du professionnalisme et de l'autorité)"
    else:
        return "Forte variabilité (indique de l'expressivité émotionnelle et de l'enthousiasme)"

# Function to analyze pace
def analyze_pace(pace):
    if pace < 2.3:
        return "Rythme lent (indiquant de la réflexion ou du sérieux)"
    elif 2.3 <= pace <= 2.8:
        return "Rythme modéré (indiquant de la confiance et de la clarté)"
    else:
        return "Rythme rapide (indiquant de l'excitation ou de l'urgence)"

# Function to analyze pauses
def analyze_pauses(pause_times):
    mean_pause = np.mean(pause_times)
    median_pause = np.median(pause_times)
    std_pause = np.std(pause_times)
    min_pause = np.min(pause_times)
    max_pause = np.max(pause_times)
    
    short_pauses = sum(0.2 <= pause <= 0.5 for pause in pause_times)
    moderate_pauses = sum(0.5 < pause <= 1.5 for pause in pause_times)
    long_pauses = sum(pause > 1.5 for pause in pause_times)
    
    analysis = {
        "Durée moyenne des pauses": mean_pause,
        "Durée médiane des pauses": median_pause,
        "Écart-type des pauses": std_pause,
        "Durée minimale des pauses": min_pause,
        "Durée maximale des pauses": max_pause,
        "Pauses courtes (0.2-0.5s)": short_pauses,
        "Pauses modérées (0.5-1.5s)": moderate_pauses,
        "Pauses longues (>1.5s)": long_pauses
    }
    
    return analysis

# Updated function with logging to vocal_log_file_path
def analyze_video(video_file, gender='male'):
    audio_file = "extracted_audio.wav"
    extract_audio(video_file, audio_file)
    
    transcript = recognize_speech(audio_file, language='fr-FR')
    log_vocal_message("Transcription: " + transcript)
    
    mean_pitch, mean_intensity, modulation = analyze_vocal_quality(audio_file)
    log_vocal_message(f"Qualité vocale - Hauteur moyenne: {mean_pitch:.2f} Hz, Intensité moyenne: {mean_intensity:.2f} dB, Modulation: {modulation:.2f} Hz")
    
    pace, pause_times = analyze_pace_and_pauses(audio_file, transcript)
    log_vocal_message(f"Rythme: {pace:.2f} mots par seconde")
    
    mean_pitch_analysis = analyze_mean_pitch(mean_pitch, gender)
    log_vocal_message(f"Analyse de la hauteur moyenne: {mean_pitch_analysis}")
    
    mean_intensity_analysis = analyze_mean_intensity(mean_intensity)
    log_vocal_message(f"Analyse de l'intensité moyenne: {mean_intensity_analysis}")
    
    modulation_analysis = analyze_modulation(modulation, gender)
    log_vocal_message(f"Analyse de la modulation: {modulation_analysis}")
    
    pace_analysis = analyze_pace(pace)
    log_vocal_message(f"Analyse du rythme: {pace_analysis}")
    
    pause_analysis = analyze_pauses(pause_times)
    log_vocal_message(f"Analyse des pauses: {pause_analysis}")

@app.route('/logs')
def get_logs():
    if os.path.exists(log_file_path):
        return send_from_directory(directory=os.path.dirname(log_file_path), path=os.path.basename(log_file_path))
    return jsonify([])

@app.route('/vocal_logs')
def get_vocal_logs():
    if os.path.exists(vocal_log_file_path):
        return send_from_directory(directory=os.path.dirname(vocal_log_file_path), path=os.path.basename(vocal_log_file_path))
    return jsonify([])

@app.route('/video_feed')
def video_feed():
    video_path = request.args.get('file_path', 'uploads/default.mp4')  # default video if none specified
    if not os.path.exists(video_path):
        return jsonify({"error": "File not found"}), 404
    return Response(generate_frames(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analyze_video', methods=['POST'])
def handle_video_analysis():
    video_file = request.args.get('file_path')
    if video_file and os.path.exists(video_file):
        analyze_video(video_file)
        return jsonify({"message": "Analysis started"}), 200
    return jsonify({"error": "Video file not found"}), 404

@app.route('/upload', methods=['POST'])
def upload_video():
    # Clear existing logs when a new file is uploaded
    open(log_file_path, 'w').close()
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        return jsonify({"file_path": file_path}), 200

@app.route('/clear_logs', methods=['POST'])
def clear_logs():
    open(log_file_path, 'w').close()
    open(vocal_log_file_path, 'w').close()
    return jsonify({"status": "success"})


def log_message(log_data):
    with open(log_file_path, "a") as log_file:
        log_file.write(f"{log_data}\n")

def log_vocal_message(log_data):
    print("Vocal log entry:", log_data)  # Debug print
    with open(vocal_log_file_path, "a") as log_file:
        log_file.write(f"{log_data}\n")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
