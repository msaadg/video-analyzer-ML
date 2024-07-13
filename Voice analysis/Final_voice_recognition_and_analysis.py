import moviepy.editor as mp
import speech_recognition as sr
import librosa
import numpy as np
import contextlib
import wave

# Function to extract audio from video
def extract_audio(video_file, audio_file):
    video = mp.VideoFileClip(video_file)
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

# Main function
def analyze_video(video_file, gender='male'):
    audio_file = "extracted_audio.wav"
    extract_audio(video_file, audio_file)
    
    transcript = recognize_speech(audio_file, language='fr-FR')
    print("Transcription:", transcript)
    
    mean_pitch, mean_intensity, modulation = analyze_vocal_quality(audio_file)
    print(f"Qualité vocale - Hauteur moyenne: {mean_pitch:.2f} Hz, Intensité moyenne: {mean_intensity:.2f} dB, Modulation: {modulation:.2f} Hz")
    
    pace, pause_times = analyze_pace_and_pauses(audio_file, transcript)
    print(f"Rythme: {pace:.2f} mots par seconde")
    
    mean_pitch_analysis = analyze_mean_pitch(mean_pitch, gender)
    print(f"Analyse de la hauteur moyenne: {mean_pitch_analysis} (valeur typique: 100-150 Hz pour les hommes, 165-225 Hz pour les femmes)")
    
    mean_intensity_analysis = analyze_mean_intensity(mean_intensity)
    print(f"Analyse de l'intensité moyenne: {mean_intensity_analysis} (valeur typique: 60-75 dB)")
    
    modulation_analysis = analyze_modulation(modulation, gender)
    print(f"Analyse de la modulation: {modulation_analysis} (valeur typique: 20-40 Hz)")
    
    pace_analysis = analyze_pace(pace)
    print(f"Analyse du rythme: {pace_analysis} (valeur typique: 2.3-2.8 mots par seconde)")
    
    pause_analysis = analyze_pauses(pause_times)
    print(f"Analyse des pauses: {pause_analysis}")

# Usage
video_file = '../titiya.mp4'
analyze_video(video_file, gender='male')
