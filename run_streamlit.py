import streamlit as st
import numpy as np
import librosa
import requests
import json
import os
import tempfile

# Constants
SCORING_URI = "https://trigger-word-detection-gqvuz.eastus.inference.ml.azure.com/score"
ENDPOINT_KEY = "ydf63eWHslFDQRwZ66sG7CkUnFQcXHrT"
TRIGGER_WORD = "boy"  # Replace with your actual trigger word


def get_auth_header():
    return {"Authorization": f"Bearer {ENDPOINT_KEY}"}


def load_audio_to_prediction(
    audio_file: str,
    sample_rate: int = 16000,
    n_mfcc: int = 13,
    max_pad_len: int = 215,
):
    """Load audio file from disk."""
    audio, sr = librosa.load(audio_file, sr=sample_rate)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs = mfccs.T
    if mfccs.shape[0] > max_pad_len:
        mfccs = mfccs[:max_pad_len, :]
    else:
        pad_width = ((0, max_pad_len - mfccs.shape[0]), (0, 0))
        mfccs = np.pad(mfccs, pad_width, mode="constant")

    # Prepare input for prediction
    mfccs = mfccs.reshape(1, max_pad_len, n_mfcc)

    print("Type of processed audio is", type(mfccs))
    return mfccs


def get_audio_duration(audio_file: str) -> float:
    """Get duration of audio file."""
    audio, sr = librosa.load(audio_file, sr=None)
    audio_duration = librosa.get_duration(y=audio, sr=sr)
    return audio_duration


def estimate_trigger_word_time(mfccs, audio_duration):
    total_frames = mfccs.shape[0]
    frame_duration = audio_duration / total_frames
    start_frame = total_frames // 3
    end_frame = 2 * total_frames // 3
    start_time = start_frame * frame_duration
    end_time = end_frame * frame_duration
    return start_time, end_time


def predict_function(audio_file, threshold=0.7):
    try:
        features = load_audio_to_prediction(audio_file)

        # Ensure the input shape matches the model's expected input
        if features.shape != (1, 215, 13):
            features = np.pad(
                features,
                ((0, 0), (0, 215 - features.shape[1]), (0, 0)),
                mode="constant",
            )

        input_data = {"input_data": {"input_layer": features.tolist()}}

        headers = get_auth_header()
        headers["Content-Type"] = "application/json"
        response = requests.post(SCORING_URI, json=input_data, headers=headers)

        st.write("Response status code:", response.status_code)
        st.write("Response content:", response.content)

        response.raise_for_status()

        prediction = response.json()

        if (
            not isinstance(prediction, list)
            or len(prediction) == 0
            or not isinstance(prediction[0], list)
        ):
            st.error(f"Unexpected prediction format. Prediction: {prediction}")
            return None

        prediction_value = prediction[0][0]
        contains_trigger_word = prediction_value > threshold

        result = {
            "filename": str(audio_file),
            "prediction": float(prediction_value),
            "contains_trigger_word": contains_trigger_word,
        }
        audio_duration = get_audio_duration(audio_file)
        if contains_trigger_word:
            start_ms, end_ms = estimate_trigger_word_time(features[0], audio_duration)
            start_seconds = int(start_ms)
            start_milliseconds = int((start_ms - start_seconds) * 1000)
            end_seconds = int(end_ms)
            end_milliseconds = int((end_ms - end_seconds) * 1000)
            result["trigger_word_time"] = (
                f"The word {TRIGGER_WORD} was detected between {start_seconds}s{start_milliseconds}ms and {end_seconds}s{end_milliseconds}ms"
            )

        return result

    except requests.exceptions.RequestException as e:
        st.error(f"Error making request to Azure ML endpoint: {str(e)}")
        if hasattr(e, "response"):
            st.error(f"Response content: {e.response.content}")
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")

    return None


def main():
    st.title("Audio Trigger Word Detection")

    uploaded_file = st.file_uploader(
        "Choose an audio file", type=["mp3", "wav", "flac"]
    )

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        if st.button("Detect Trigger Word"):
            result = predict_function(audio_file=tmp_file_path)

            if result:
                st.write(f"Filename: {uploaded_file.name}")
                st.write(f"Prediction: {result['prediction']:.4f}")
                st.write(f"Contains trigger word: {result['contains_trigger_word']}")
                if result["contains_trigger_word"]:
                    st.write(result["trigger_word_time"])
            else:
                st.error(
                    "Failed to process the audio. Please check the error messages above."
                )

        os.unlink(tmp_file_path)


if __name__ == "__main__":
    main()
