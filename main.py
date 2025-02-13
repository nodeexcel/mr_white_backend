from fastapi import File, UploadFile
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import openai
import json
import logging
import os
import edge_tts
import librosa
from g2p_en import G2p
from fastapi.staticfiles import StaticFiles
import speech_recognition as sr
from pydub import AudioSegment
import uuid
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()
app.mount("/backend/audio", StaticFiles(directory="audio"), name="audio")


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing; adjust as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# data models
class ChatRequest(BaseModel):
    message: str


""" ******** Helper Function **********"""
# def generate_response(user_text):
    # """
    #     generate response using open source Ollama model llama3
    # """

    # messages = [
    #     (
    #         "system",
    #         "You are a helpful assistant",
    #     ),
    #     ("human", user_text),
    # ]

    # llm = ChatOllama(
    #     model="llama3",
    #     temperature=0,
    # )
    # # for chunk in llm.stream(messages):
    # #     return chunk
    # ai_msg = llm.invoke(messages)
    # return str(ai_msg.content)


def generate_response(user_text: str):
    """
        response using open ai  
    """
    
    try:
    
        completion = client.chat.completions.create(
            model="chatgpt-4o-latest",
            stream=True,
            temperature=0,
            messages=[
                {"role": "system", "content": "you are helpful assistent"},
                {
                    "role": "user",
                    "content": user_text
                }
            ]
        )
        
        # return str(completion.choices[0].message.content)
        for chunk in completion:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
                
    except Exception as e:
        logger.error("Exception occurred in generate_response: %s", e, exc_info=True)
        yield f"[Error generating response: {str(e)}]"


async def text_to_speech(text: str):
    voice = "en-US-AriaNeural"
    filename = f"{uuid.uuid4()}.mp3"
    output_path = f"./audio/{filename}"

    if not os.path.exists("audio"):
        os.makedirs("audio")

    tts = edge_tts.Communicate(text, voice)  # Only pass text & voice
    await tts.save(output_path)  # Save audio to file

    return output_path


def extract_phonemes(text: str):
    """
        function to extract the phonemes
    """
    g2p = G2p()
    phonemes = g2p(text)
    return phonemes


def extract_phonemes_with_timings(audio_path: str, text: str):
    """
    Uses librosa to detect onset times from the TTS audio and maps those
    to phonemes extracted from the text.
    """
    # extract phones text
    phonemes = extract_phonemes(text)
    if not phonemes:
        return []

    # load audio
    y, sr = librosa.load(audio_path, sr=None)
    total_duration = librosa.get_duration(y=y, sr=sr)

    # This returns an array of times (in seconds) where onsets are detected.
    onset_times = librosa.onset.onset_detect(y=y, sr=sr, units="time")

    phoneme_timings = []

    # Map detected onsets to phonemes.
    if len(onset_times) >= len(phonemes):
        # If we have as many (or more) onsets as phonemes,
        for i, phoneme in enumerate(phonemes):
            # For each phoneme, the start time is the detected onset.
            start = onset_times[i]
            # The end time is the next onset, or the total duration for the last phoneme.
            end = onset_times[i + 1] if i + 1 < len(onset_times) else total_duration
            phoneme_timings.append({"phoneme": phoneme, "start": start, "end": end})
    else:
        # If there are not enough onsets detected, fallback to equal division.
        time_per_phoneme = total_duration / len(phonemes)
        current_time = 0.0
        for phoneme in phonemes:
            phoneme_timings.append(
                {
                    "phoneme": phoneme,
                    "start": current_time,
                    "end": current_time + time_per_phoneme,
                }
            )
            current_time += time_per_phoneme

    return phoneme_timings


def convert_audio_to_wav(input_path, output_path):
    """Convert MP3 (or other formats) to WAV."""
    # If needed, explicitly specify the format (e.g., format="mp3")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"The input file {input_path} does not exist.")
    audio = AudioSegment.from_file(input_path, format='mp3')

    audio.export(output_path, format="wav")


def speech_to_text(audio_path):
    """Convert audio file to text using SpeechRecognition."""
    
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data)
        
    except Exception as e:
        print("except as ", e)
        logger.error("Error in speech_to_text: %s", e)
        return f"An error occurred: {str(e)}"



@app.post("/api/chat")
async def handle_chat(request: ChatRequest):
    """
        api to handle the chat responses
    """
    user_text = request.message
    
    audio_path=None
    phonemes_timing=None
    
    # Generate AI response
    result = generate_response(user_text=user_text)

    # Convert AI response to speech
    # uncomment to get phonemes and their audio timing
    # audio_path = await text_to_speech(result)
    # phonemes_timing = extract_phonemes_with_timings(audio_path=audio_path, text=result)
    
    return {"response": result, "audio_url": audio_path, "phonemes": phonemes_timing}


@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        
        while True:
            user_text = await websocket.receive_text()
            # Stream each delta chunk to the client.
            for delta in generate_response(user_text):
                await websocket.send_text(json.dumps({"delta": delta}))
            # Optionally, send a final message to indicate completion.
            await websocket.send_text(json.dumps({"complete": True}))
    except WebSocketDisconnect:
        print("WebSocket disconnected")
        logger.info("WebSocket disconnected")


@app.post("/process_audio/")
async def process_audio(file: UploadFile = File(...)):
    os.makedirs("./temp_audio", exist_ok=True)
    input_audio_path = f"./temp_audio/{file.filename}"

    with open(input_audio_path, "wb") as buffer:
        buffer.write(file.file.read())

    # Convert to WAV if not already in WAV format
    output_audio_path = input_audio_path.rsplit(".", 1)[0] + ".wav"
    if file.filename.lower().endswith(".mp3"):
        convert_audio_to_wav(input_audio_path, output_audio_path)
        os.remove(input_audio_path)  # Delete original MP3
    else:
        output_audio_path = input_audio_path  # Already WAV

    # Perform speech-to-text
    try:
        print("output_audio_path", output_audio_path)
        text = speech_to_text(output_audio_path)
        os.remove(output_audio_path)  # Clean up
        return {"text": text}
    except Exception as e:
        return {"error": str(e)}



if __name__ == "__main__":
    # Runs the app with uvicorn directly
    uvicorn.run("main:app", host="116.202.210.102", port=5007, reload=True)