import tkinter as tk
from tkinter import filedialog, messagebox
from emotion_recognition import predict_emotion
import sounddevice as sd
import soundfile as sf
import tempfile
import os

DURATION = 3  # seconds
SAMPLE_RATE = 22050

class EmotionRecognitionApp:
    def __init__(self, master):
        self.master = master
        master.title('Speech Emotion Recognition')
        master.geometry('400x250')
        self.file_path = None

        self.label = tk.Label(master, text='Select a .wav file or record to predict emotion:', font=('Arial', 12))
        self.label.pack(pady=10)

        self.select_button = tk.Button(master, text='Browse', command=self.browse_file)
        self.select_button.pack(pady=5)

        self.record_button = tk.Button(master, text='Record & Predict', command=self.record_and_predict)
        self.record_button.pack(pady=5)

        self.predict_button = tk.Button(master, text='Predict Emotion', command=self.predict, state=tk.DISABLED)
        self.predict_button.pack(pady=5)

        self.result_label = tk.Label(master, text='', font=('Arial', 14, 'bold'))
        self.result_label.pack(pady=10)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[('WAV files', '*.wav')])
        if file_path:
            self.file_path = file_path
            self.result_label.config(text='')
            self.predict_button.config(state=tk.NORMAL)

    def predict(self):
        if not self.file_path:
            messagebox.showwarning('No file', 'Please select a .wav file first.')
            return
        try:
            emotion = predict_emotion(self.file_path, model_path='emotion_model_rf.joblib', le_path='label_encoder_rf.joblib')
            self.result_label.config(text=f'Predicted Emotion: {emotion}')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to predict emotion.\n{e}')

    def record_and_predict(self):
        self.result_label.config(text='Recording...')
        self.master.update()
        try:
            audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmpfile:
                tmpfile_path = tmpfile.name
            sf.write(tmpfile_path, audio, SAMPLE_RATE)
            emotion = predict_emotion(tmpfile_path, model_path='emotion_model_rf.joblib', le_path='label_encoder_rf.joblib')
            self.result_label.config(text=f'Predicted Emotion: {emotion}')
            os.unlink(tmpfile_path)
        except Exception as e:
            messagebox.showerror('Error', f'Failed to record or predict emotion.\n{e}')

if __name__ == '__main__':
    root = tk.Tk()
    app = EmotionRecognitionApp(root)
    root.mainloop() 