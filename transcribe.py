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

system_message = '''Input: Audio file of a podcast with Japanese language and multiple speakers.
Output Format: A properly formatted .srt file, including speaker identification for each line of dialogue.

Guidelines:
1.  **Transcription Accuracy:** Transcribe all spoken Japanese dialogue with high accuracy.
2.  **Speaker Diarization and Naming:**
    *   Perform speaker diarization to accurately identify and differentiate between individual speakers.
    *   For each line of dialogue, prepend the speaker's name or label.
    *   **Speaker Name Format:** Use the format `[Speaker Name]:` at the beginning of their respective dialogue lines (e.g., `[Host]: こんにちは`).
    *   **Speaker Identification:**
        *   If specific names are explicitly mentioned or clearly inferable from the audio context (e.g., a speaker introduces themselves or is addressed by name), use the identified name (e.g., `[Tsukimura Temari]`, `[Saki]`).
        *   If names are not identifiable, assign consistent, descriptive labels (e.g., `[Speaker 1]`, `[Speaker 2]`, `[Host]`, `[Guest]`) throughout the entire transcript.
3.  **Subtitle Formatting and Timing (Industry Standards):**
    *   **Segmentation:** Each subtitle entry should represent a coherent thought or sentence. Break subtitles at natural pauses, sentence endings, or logical clause boundaries. Avoid splitting words across lines or subtitle entries.
    *   **Line Length:** Limit each line of text (excluding the speaker name prefix) to a maximum of approximately 42 characters (including spaces and punctuation).
    *   **Lines per Subtitle Block:** Each subtitle block should contain a maximum of two lines.
    *   **Reading Speed:** Target a reading speed of 15-18 characters per second (CPS) to ensure comfortable readability. Adjust timings accordingly.
    *   **Minimum Display Duration:** Each subtitle block should be displayed for a minimum of 1.5 seconds.
    *   **Maximum Display Duration:** No single subtitle block should remain on screen for longer than 7 seconds.
    *   **Timecodes:** Ensure precise start and end timecodes for each subtitle block.
    *   **Markdown:** Do not use Markdown or backticks in your output.
4.  **Non-Speech Elements:**
    *   Include descriptive labels in brackets for significant non-speech audio events (e.g., `[Music]`, `[Laughter]`, `[Applause]`, `[Silence]`) where relevant to the context.
    *   For dialogue that is unclear or unintelligible, use `[unintelligible]`.
5. **Translation:** 
    * If the user requests a translation, the output should be in the target language.
    * If the target language is not provided, assume the user wants English translation.
    * If the user does not request a translation, do not provide it. 
    * Regardless, the output should always be properly formatted and include speaker identification.'''

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

def get_warning_message(con_count: int = 0, task_type: str = "Transcription"):
        msg = f'''
        {'\n' + '=' * 24}
        \nWARNING: {task_type} may be incomplete, as a token limit was reached {con_count} time(s)
        \n{'=' * 24}
        '''
        return msg

def is_response_truncated(finish_reason: str = ""):
    """
    Check the finish reason to detect possible truncation.
    """
    truncated_finish_reasons = ['MAX_TOKENS', 'SAFETY', 'BLOCKLIST', 'PROHIBITED_CONTENT']
    if finish_reason in truncated_finish_reasons:
        return True
    else:
        return False

async def get_complete_response(messages, max_continuations=10):
    """
    Detection of truncation and automatic continuation.
    """
    complete_response = ""
    continuation_count = 0
    
    while continuation_count <= max_continuations:
        print(f"\n--- Getting response (attempt {continuation_count + 1}) ---")
        
        current_response = ""
        finish_reason = ""
        async for chunk in model.astream(messages):
            if hasattr(chunk, 'content') and chunk.content:
                print(chunk.content, end="", flush=True)
                current_response += chunk.content
            if hasattr(chunk, 'response_metadata') and chunk.response_metadata:
                if 'finish_reason' in chunk.response_metadata:
                    finish_reason = chunk.response_metadata['finish_reason']

        complete_response += current_response

        # Check the response finish reason for eventual continuation.
        if not is_response_truncated(finish_reason) or continuation_count >= max_continuations:
            break

        # Add continuation prompt
        print(f"\n\n[Detected truncated response, requesting continuation...]")
        continuation_count += 1

        # Add the current response to conversation history and request continuation
        messages.append(HumanMessage(content=current_response))
        messages.append(HumanMessage(content="Continue from where you left off. Continue the transcription/translation exactly where it was cut off."))

    if continuation_count > max_continuations:
        print(f"\n[Warning: Reached maximum continuation attempts ({max_continuations}). Response may still be incomplete.]")

    return complete_response, continuation_count



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

        # Get a complete response with automatic continuation
        full_response, con_count = await get_complete_response(messages.copy())
        if con_count > 0:
            print(get_warning_message(con_count))
        print(f"\nTranscription complete!")

        await save_to_srt(full_response, filename)

        if translate:
            print("\nStarting translation...")
            query2 = f"Translate the above content to {language}, output a proper .srt file."
            translation_messages = messages + [HumanMessage(content=query2)]

            print("Translation: ")
            
            # Get a complete translation response with automatic continuation
            translation_response, con_count = await get_complete_response(translation_messages)
            if con_count > 0:
                print(get_warning_message(con_count, "Translation"))
            print(f"\nTranslation complete!")
            await save_to_srt(translation_response, f"{filename}_{lang_suffix}")
        else:
            print("\nTranslation skipped")

    except Exception as e:
        print("An error occurred during transcription: ", e)

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