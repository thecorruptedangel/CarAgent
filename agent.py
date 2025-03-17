import pyaudio
import asyncio
import json
import websockets
import shutil
import numpy as np
from dotenv import load_dotenv
import pandas as pd
from groq import Groq
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import time
from openai import OpenAI
import io
import pygame
import re
import wave, subprocess
load_dotenv()
pygame.mixer.init()

# Load environment variables
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 8000
DEEPGRAM_URL = f"wss://api.deepgram.com/v1/listen?model=nova-2-conversationalai&utt_split=1.5&punctuate=true&filler_words=true&encoding=linear16&sample_rate={RATE}"

# Set the system prompt
system_prompt = {
    "role": "system",
    "content": os.environ.get("INSTRUCT")
}
chat_history = [system_prompt]

def is_installed(lib_name: str) -> bool:
    lib = shutil.which(lib_name)
    return lib is not None

def check_call_end(text):
    if '[CALL_END]' in text:
        append_appointment(text)
        pygame.mixer.quit()
        quit()


def append_appointment(data_string):
    match = re.search(r'\[(.*?)\]', data_string)
    if not match:
        return
    
    data = match.group(1)
    
    # Adjust regex to handle numbers with commas correctly
    field_value_pairs = re.findall(r'([^:,\[\]\n]+?):\s*([^\[\]\n]+?(?:,\d{3})*)', data)

    if not field_value_pairs:
        return

    headers, values = zip(*[(field.strip(), value.strip()) for field, value in field_value_pairs])

    df = pd.DataFrame([values], columns=headers)

    file_name = 'appointments.xlsx'
    if os.path.isfile(file_name):
        with pd.ExcelWriter(file_name, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            df.to_excel(writer, sheet_name='Sheet1', index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
    else:
        df.to_excel(file_name, index=False)


def text2wav(phrase, model_name):
    try:
        process = subprocess.Popen(
            ['.\piper\piper.exe', '--model', model_name, '--output-raw'], 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE,  # Capture stdout
            stderr=subprocess.PIPE   # Capture stderr
        )
        output, _ = process.communicate(input=phrase.encode(), timeout=10)
        
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setparams((1, 2, 22050, 0, 'NONE', 'not compressed'))
            wav_file.writeframes(output)
        
        wav_buffer.seek(0)
        pygame.mixer.music.load(wav_buffer)
        print("Assistant:", phrase)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def speak(input_text):
    client = OpenAI(api_key=OPENAI_API_KEY)
    audio_data = io.BytesIO()
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=input_text,
    ) as response:
        for chunk in response.iter_bytes():
            audio_data.write(chunk)

    audio_data.seek(0)
    pygame.mixer.music.load(audio_data)
    print("Assistant:", input_text)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(5)

def generate_white_noise(duration_ms, sample_rate=RATE):
    num_samples = int(duration_ms * sample_rate / 1000)
    return np.random.randn(num_samples).astype(np.float32)

def chat_with_llm(llm_input, chat_history):
    try:
        chat_history.append({"role": "user", "content": llm_input})
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=chat_history,
            max_tokens=500,
            temperature=0
        )
        chat_history.append({
            "role": "assistant",
            "content": response.choices[0].message.content
        })
        return response.choices[0].message.content
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

async def listen_and_process():
    llm_output = chat_with_llm("Hello", chat_history)
    #speak(llm_output)
    text2wav(llm_output, model_path)
    await asyncio.sleep(0.05)
    audio_queue = asyncio.Queue()
    shutdown_event = asyncio.Event()
    mic_active = asyncio.Event()
    mic_active.set()
    last_tts_time = 0

    def mic_callback(input_data, frame_count, time_info, status_flag):
        current_time = time.time()
        if not shutdown_event.is_set() and mic_active.is_set() and (current_time - last_tts_time) > 1:
            audio_queue.put_nowait(input_data)
        return (input_data, pyaudio.paContinue)

    async def sender(ws):
        try:
            while not shutdown_event.is_set():
                if mic_active.is_set():
                    try:
                        mic_data = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        mic_data = None
                else:
                    await asyncio.sleep(0.5)
                    white_noise = generate_white_noise(CHUNK / RATE * 1000)
                    mic_data = white_noise.tobytes()
                
                if mic_data:
                    await ws.send(mic_data)
        except Exception as e:
            await asyncio.sleep(0.1)

    async def receiver(ws):
        global chat_history
        try:
            async for msg in ws:
                if shutdown_event.is_set():
                    break
                res = json.loads(msg)
                try:
                    if res.get("is_final"):
                        transcript = res.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
                        if transcript:
                            print("You:", transcript)

                            try:
                                mic_active.clear()  # Deactivate microphone
                                llm_output = chat_with_llm(transcript, chat_history)
                                # speak(llm_output)
                                text2wav(llm_output, model_path)
                                check_call_end(llm_output)
                                await asyncio.sleep(0.05)
                                mic_active.set()  # Reactivate microphone
                            except Exception as e:
                                print(f"Error: {str(e)}")
                except KeyError:
                    await asyncio.sleep(0.1)
                    print("key-errror")
        except websockets.exceptions.ConnectionClosedError as e:
            print("Speak ..")
            await asyncio.sleep(0.1)


    async def microphone():
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, stream_callback=mic_callback)
        stream.start_stream()

        try:
            while not shutdown_event.is_set():
                await asyncio.sleep(0.1)
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

    while not shutdown_event.is_set():
        try:
            async with websockets.connect(DEEPGRAM_URL, extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"}) as ws:
                sender_task = asyncio.create_task(sender(ws))
                receiver_task = asyncio.create_task(receiver(ws))
                microphone_task = asyncio.create_task(microphone())
                done, pending = await asyncio.wait(
                    [sender_task, receiver_task, microphone_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                for task in pending:
                    task.cancel()

                await asyncio.gather(*pending, return_exceptions=True)

        except asyncio.CancelledError:
            break
        except websockets.exceptions.ConnectionClosedError as e:
            print("Speak again..2")
            await asyncio.sleep(0.1)  # Wait before retrying
        except Exception as e:
            print("Speak again..3")
            await asyncio.sleep(0.1)  # Wait before retrying
            

    shutdown_event.set()

def main():
    asyncio.run(listen_and_process())

#model_path = r".\piper\voice_model\en_US-ljspeech-medium.onnx"
# model_path = r".\piper\voice_model\en_US-libritts_r-medium.onnx"
model_path = r".\piper\voice_model\en_US-bryce-medium.onnx"

if __name__ == "__main__":
    main()