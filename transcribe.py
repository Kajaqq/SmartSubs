from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys

from dotenv import load_dotenv
from google import genai
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

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
    *   **Speaker Name Format:** 
        *   Use the format `[Speaker Name]:` at the beginning of their respective dialogue lines (e.g., `[Host]: こんにちは`).
        *   If a subtitle block contains two lines, add the Speaker Name only in the first one.
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

def AudioMessage(audio_uri: str, mime_type: str):
    return HumanMessage(content=[{"type": "media", "file_uri": audio_uri, "mime_type": mime_type}])

# noinspection PyCompatibility
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

def prepare_file(file_path: str):
    if not detect_video(file_path):
        return file_path  # Return the original file for non-video files
    input_video = file_path
    output_audio = get_output_path(input_video)
    print(f"Video detected, extracting audio to '{output_audio}'...")
    extract_audio(input_video, output_audio)
    return output_audio

async def upload_audio(audio_file: str | pathlib.Path[str]):
    client = genai.Client()
    print("Uploading audio file...")
    myfile = await client.aio.files.upload(file=audio_file)
    file_name = myfile.name
    myfile_data = await client.aio.files.get(name=file_name)
    audio_uri = myfile_data.uri
    mime_type = myfile_data.mime_type
    print(f"Audio file uploaded successfully: {audio_uri}\n")
    audio_message = AudioMessage(audio_uri, mime_type)
    return audio_message

async def save_to_srt(transcript_data, filename: pathlib.Path[str], task_type='Transcript'):
    filename = filename.with_suffix(".srt")
    with open(filename, "w", encoding='utf-8') as f:
        if isinstance(transcript_data, str):
            f.write(transcript_data)
        else:
            for line in transcript_data:
                f.write(line)
    print(f"{task_type} saved to {filename}")
    print('=' * 24)

async def get_complete_response(messages: list, max_continuations=10):
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
            messages.append(AIMessage(content=complete_response))
            break

        # Add continuation prompt
        print(f"\n\n[Detected truncated response, requesting continuation...]")
        continuation_count += 1

        # Add the current response to the conversation history and request continuation
        messages.append(AIMessage(content=current_response))
        messages.append(HumanMessage(
            content="Continue from where you left off. Continue the transcription/translation exactly where it was cut off."))

    if continuation_count > max_continuations:
        print(
            f"\n[Warning: Reached maximum continuation attempts ({max_continuations}). Response may still be incomplete.]")

    return messages, continuation_count

async def transcribe(filename: pathlib.Path[str], sys_msg=system_message):
    sys_msg = SystemMessage(content=sys_msg)
    audio_msg = await upload_audio(filename)
    messages = [sys_msg,audio_msg]

    print('=' * 24)
    print('Starting transcription...')
    print('=' * 24)

    # Get a complete response with automatic continuation in case of a max token error
    messages, con_count = await get_complete_response(messages.copy())
    full_response = messages[-1].content

    # Warn the user if a continuation prompt was needed
    if con_count > 0:
        print(get_warning_message(con_count))

    print('\n\n' + '=' * 24)
    print(f"Transcription complete!")

    await save_to_srt(full_response, filename)

    return messages

async def translate(filename: pathlib.Path[str], language: str, messages: list):
    lang_suffix = language.lower().replace(' ', '_')
    output_filename = pathlib.Path(filename.stem + f"_{lang_suffix}")
    trans_prompt = f"Translate the above content to {language}, output a proper .srt file."
    messages.append(HumanMessage(content=trans_prompt))

    print()
    print("\nStarting translation...")

    # Get a complete response with automatic continuation in case of a max token error
    messages, con_count = await get_complete_response(messages)
    translation_response = messages[-1].content

    # Warn the user if a continuation prompt was needed
    if con_count > 0:
        print(get_warning_message(con_count, "Translation"))

    print('\n' + '=' * 24)
    print(f"\nTranslation complete!")

    await save_to_srt(translation_response, output_filename, 'Translation')

if __name__ == "__main__":
    async def main(file, trans_flag, language):
        file = prepare_file(file)
        file_path = pathlib.Path(file)
        messages = await transcribe(file_path)
        if trans_flag:
            await translate(file_path, language, messages)
        else:
            print("Skipping translation due to --no-translate flag.\n\n")

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
    asyncio.run(main(args.file, args.translate, args.language))
