
import asyncio
import pathlib
import sys
from dotenv import load_dotenv
from google import genai
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from video_split import detect_video, get_output_path, extract_audio

########################################################
model_name = 'gemini-2.5-flash'

system_message = '''Input: Podcast with Japanese language and multiple speakers.
Output Format: proper .srt file with speaker names
Guidelines: Analyze the audio file, detect the speaker names, and transcribe. Follow industry standards for subtitle length.'''

########################################################

load_dotenv()
model = init_chat_model(model=model_name, model_provider="google_genai", temperature=0.20)

def prepare_file(file_path):
    if not detect_video(file_path):
        return file_path # Return the original file for non-video files
    input_video = file_path
    output_audio = get_output_path(input_video)
    print(f"Video detected, extracting audio to '{output_audio}'...")
    extract_audio(input_video, output_audio)
    return output_audio

async def upload_audio(audio_file):
    client = genai.Client()
    print("Uploading audio file...")
    myfile = await client.aio.files.upload(file=audio_file)
    file_name = myfile.name
    myfile_data = await client.aio.files.get(name=file_name)
    audio_uri = myfile_data.uri
    mime_type = myfile_data.mime_type
    print(f"Audio file uploaded successfully: {audio_uri}")
    audio_message = HumanMessage(content=[{"type": "media", "file_uri": audio_uri, "mime_type": mime_type}])
    return audio_message

async def save_to_srt(transcript_data, filename="transcript"):
    filename = filename + '.srt'
    with open(filename, "w", encoding='utf-8') as f:
        if isinstance(transcript_data, str):
            f.write(transcript_data)
        else:
            for line in transcript_data:
                f.write(line)
    print(f"Transcript saved to {filename}")

async def transcribe(audio_file, translate, language, sys_msg=system_message):
    filename = pathlib.Path(audio_file).stem
    lang_suffix = language.lower().replace(' ', '_')
    try:
        audio_file = prepare_file(audio_file)
        audio_msg = await upload_audio(audio_file)
        messages = [
            SystemMessage(content=sys_msg),
            audio_msg
        ]

        print('Starting transcription...')
        print("Transcription: ")

        full_response = ""

        async for chunk in model.astream(messages):
            if hasattr(chunk, 'content') and chunk.content:
                print(chunk.content, end="", flush=True)
                full_response += chunk.content

        print(f"\nTranscription complete!")
        await save_to_srt(full_response, filename)

        if translate:
            print("\nStarting translation...")
            query2 = f"Translate the above content to {language}, output a proper .srt file."
            messages.append(HumanMessage(content=query2))

            print("Translation: ")
            translation_response = ""

            async for chunk in model.astream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    print(chunk.content, end="", flush=True)
                    translation_response += chunk.content

            print(f"\nTranslation complete!")
            await save_to_srt(translation_response, f"{filename}_{lang_suffix}")
        else:
            print("\nTranslation skipped")

    except Exception as e:
        print(f"Error during transcription: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Transcribe audio files and translate to specified language')
    parser.add_argument('file', type=str, help='Path to the audio file to transcribe')
    parser.add_argument('--no-translate', '-n', dest='translate', action='store_false',
                        help='Disable translation (translation is enabled by default)')
    parser.add_argument('--language', '-l', type=str, default='English',
                        help='Target language for translation (default: English)')

    parser.set_defaults(translate=True)
    # Parse arguments
    args = parser.parse_args()
    # Validate file path
    if not pathlib.Path(args.file).exists():
        print(f"Error: File '{args.file}' does not exist.")
        sys.exit(1)

    # Run transcription with provided arguments
    asyncio.run(transcribe(args.file, args.translate, args.language))