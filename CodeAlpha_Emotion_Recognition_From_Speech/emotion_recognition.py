import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

# Emotion labels for RAVDESS (based on filename convention)
EMOTION_LABELS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled

def parse_emotion_from_filename(filename):
    parts = filename.split('-')
    if len(parts) > 2:
        emotion_code = parts[2]
        return EMOTION_LABELS.get(emotion_code, 'unknown')
    return 'unknown'

def load_data(dataset_path):
    features = []
    labels = []
    for actor_folder in os.listdir(dataset_path):
        actor_path = os.path.join(dataset_path, actor_folder)
        if not os.path.isdir(actor_path):
            continue
        for file in os.listdir(actor_path):
            if file.endswith('.wav'):
                file_path = os.path.join(actor_path, file)
                emotion = parse_emotion_from_filename(file)
                if emotion == 'unknown':
                    continue
                mfccs = extract_features(file_path)
                features.append(mfccs)
                labels.append(emotion)
    return np.array(features), np.array(labels)

def main():
    print('Loading data...')
    X, y = load_data('dataset')
    print(f'Loaded {len(X)} samples.')
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    print('Training RandomForest model...')
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Test accuracy: {acc:.2f}')
    # Save model and label encoder
    joblib.dump(clf, 'emotion_model_rf.joblib')
    joblib.dump(le, 'label_encoder_rf.joblib')

def predict_emotion(file_path, model_path='emotion_model_rf.joblib', le_path='label_encoder_rf.joblib'):
    clf = joblib.load(model_path)
    le = joblib.load(le_path)
    mfccs = extract_features(file_path)
    mfccs = np.expand_dims(mfccs, axis=0)
    pred = clf.predict(mfccs)
    emotion = le.inverse_transform(pred)[0]
    return emotion

if __name__ == '__main__':
    main() 