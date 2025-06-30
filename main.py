from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys
from dataclasses import dataclass
from typing import List, Tuple

from dotenv import load_dotenv
from google import genai
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from video_split import detect_video, get_output_path, extract_audio


@dataclass
class TranscriptionConfig:
    """Configuration for audio transcription and translation."""
    # Change the model settings here
    model_name: str = 'gemini-2.5-flash'
    temperature: float = 0.20
    max_continuations: int = 10
    separator : str = '='* 24
    merge_sign: str = '\n' + separator + 'MERGED' + separator + '\n'

class Prompts:
    """Container for system prompts and messages."""
    
    DEFAULT_SYSTEM_MESSAGE = '''Input: Audio file of a podcast with Japanese language and multiple speakers.
Output Format: A properly formatted .srt file, including speaker identification for each line of dialogue.

Guidelines:
1.  **Transcription Accuracy:** Transcribe all spoken Japanese dialogue with high accuracy.
2.  **Speaker Diarization and Naming:**
    *   Perform speaker diarization to accurately identify and differentiate between individual speakers.
    *   For each line of dialogue, prepend the speaker's name or label.
    *   **Speaker Name Format:** 
        *   Use the format `[Speaker Name]:` at the beginning of their respective dialogue lines (e.g., `[Host]: こんにちは`).
        *   If a subtitle block contains multiple lines, add the Speaker Name only in the first one.
    *   **Speaker Identification:**
        *   If specific names are explicitly mentioned or clearly inferable from the audio context (e.g., a speaker introduces themselves or is addressed by name), use the identified name (e.g., `[Tsukimura Temari]`, `[Saki]`).
        *   If names are not identifiable, assign consistent, descriptive labels (e.g., `[Speaker 1]`, `[Speaker 2]`, `[Host]`, `[Guest]`) throughout the entire transcript.
3.  **Subtitle Formatting and Timing (Industry Standards):**
    *   **Segmentation:** Each subtitle entry should represent a coherent thought or sentence. Break subtitles at natural pauses, sentence endings, or logical clause boundaries. Avoid splitting words across lines or subtitle entries.
    *   **Line Length:** For English language, limit each line of text (excluding the speaker name prefix) to a maximum of approximately 42 characters (including spaces and punctuation).
    *   **Lines per Subtitle Block:** Each subtitle block should contain a maximum of two lines.
    *   **Reading Speed:** For English language, target a reading speed of 15-18 characters per second (CPS) to ensure comfortable readability. Adjust timings accordingly.
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

    DEFAULT_CONTINUE_MESSAGE = "Continue from where you left off. Continue the transcription/translation exactly where it was cut off."

    DEFAULT_TRANSLATION_PROMPT = "Translate the above content to {target_language}, output a proper .srt file."

class AudioTranscriber:
    """Main class for audio transcription and translation functionality."""
    
    TRUNCATED_FINISH_REASONS = ['MAX_TOKENS', 'SAFETY', 'BLOCKLIST', 'PROHIBITED_CONTENT']
    
    def __init__(self, config: TranscriptionConfig, disable_merge_sign: bool = False):
        self.config = config
        self.disable_merge_sign = disable_merge_sign
        self.model = self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the language model."""
        load_dotenv()
        return init_chat_model(
            model=self.config.model_name,
            model_provider="google_genai",
            temperature=self.config.temperature
        )
    
    @staticmethod
    def create_audio_message(audio_uri: str, mime_type: str) -> HumanMessage:
        """Create an audio message for the model."""
        return HumanMessage(content=[{
            "type": "media",
            "file_uri": audio_uri,
            "mime_type": mime_type
        }])


    def get_warning_message(self, continuation_count: int, task_type: str = "Transcription") -> str:
        """Generate a warning message for incomplete transcriptions."""
        merge_sign_info = (
            '\nFor your convenience, a merge sign has been added to the output file '
            'to indicate where the continuation prompt was applied.'
            if not self.disable_merge_sign else ''
        )
        
        return f'''
        \n{self.config.separator}
        \nWARNING: {task_type} may be incomplete, as a token limit was reached {continuation_count} time(s)
        {merge_sign_info}
        \n{self.config.separator}
        '''
    
    @staticmethod
    def is_response_truncated(finish_reason: str) -> bool:
        """Check if the response was truncated based on the finish reason."""
        return finish_reason in AudioTranscriber.TRUNCATED_FINISH_REASONS
    
    @staticmethod
    def prepare_file(file_path: str) -> str:
        """Prepare the input file, extracting audio from video if necessary."""
        if not detect_video(file_path):
            return file_path
        
        output_audio = get_output_path(file_path)
        print(f"Video detected, extracting audio to '{output_audio}'...")
        extract_audio(file_path, output_audio)
        return output_audio
    
    async def upload_audio(self, audio_file: str | pathlib.Path) -> HumanMessage:
        """Upload an audio file and return an audio message."""
        client = genai.Client()
        print("Uploading audio file...")
        
        uploaded_file = await client.aio.files.upload(file=audio_file)
        file_data = await client.aio.files.get(name=uploaded_file.name)
        
        print(f"Audio file uploaded successfully: {file_data.uri}\n")
        return self.create_audio_message(file_data.uri, file_data.mime_type)
    
    async def save_to_srt(self, transcript_data: str, filename: pathlib.Path, task_type: str = 'Transcript') -> None:
        """Save transcript data to an SRT file."""
        output_filename = filename.with_suffix(".srt")
        
        with open(output_filename, "w", encoding='utf-8') as f:
            f.write(transcript_data)
        
        print(f"{task_type} saved to {output_filename}")
        print(self.config.separator)
    
    async def get_complete_response(self, messages: List) -> Tuple[List, int]:
        """Get a complete response with automatic continuation handling."""
        complete_response = ""
        continuation_count = 0
        merge_sign = "" if self.disable_merge_sign else self.config.merge_sign
        
        while continuation_count <= self.config.max_continuations:
            if continuation_count == 0:
                print(f"\n--- Getting response ---")
            else:
                print(f"\n--- Getting response (attempt {continuation_count + 1}) ---")
            
            current_response = ""
            finish_reason = ""

            async for chunk in self.model.astream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    print(chunk.content, end="", flush=True)
                    current_response += chunk.content
                if hasattr(chunk, 'response_metadata') and chunk.response_metadata:
                    if 'finish_reason' in chunk.response_metadata:
                        finish_reason = chunk.response_metadata['finish_reason']
            
            complete_response += merge_sign + current_response
            
            # Check if the response is complete
            if not self.is_response_truncated(finish_reason) or continuation_count >= self.config.max_continuations:
                messages.append(AIMessage(content=complete_response))
                break
            
            # Handle continuation
            print(f"\n\n[Detected truncated response, requesting continuation...]")
            continuation_count += 1
            
            messages.append(AIMessage(content=current_response))
            messages.append(HumanMessage(
                content=Prompts.DEFAULT_CONTINUE_MESSAGE
            ))
        
        if continuation_count > self.config.max_continuations:
            print(f"\n[Warning: Reached maximum continuation attempts ({self.config.max_continuations}). "
                  f"Response may still be incomplete.]")
        
        return messages, continuation_count
    
    async def transcribe(self, filename: pathlib.Path, system_message: str = None) -> List:
        """Transcribe the audio file to SRT format."""
        if system_message is None:
            system_message = Prompts.DEFAULT_SYSTEM_MESSAGE
        separator = self.config.separator
        
        sys_msg = SystemMessage(content=system_message)
        audio_msg = await self.upload_audio(filename)
        messages = [sys_msg, audio_msg]
        
        print(separator)
        print('Starting transcription...')
        print(separator)
        
        messages, continuation_count = await self.get_complete_response(messages.copy())
        full_response = messages[-1].content
        
        if continuation_count > 0:
            print(self.get_warning_message(continuation_count))
        
        print('\n' + separator)
        print("Transcription complete!")
        
        await self.save_to_srt(full_response, filename)
        return messages
    
    async def translate(self, filename: pathlib.Path, messages: List, target_language: str) -> None:
        """Translate transcribed content to the target language."""
        lang_suffix = target_language.lower().replace(' ', '_')
        output_filename = pathlib.Path(filename.stem + f"_{lang_suffix}")
        
        translation_prompt = Prompts.DEFAULT_TRANSLATION_PROMPT.format(target_language=target_language)
        messages.append(HumanMessage(content=translation_prompt))
        
        print("\nStarting translation...")
        
        messages, continuation_count = await self.get_complete_response(messages)
        translation_response = messages[-1].content
        
        if continuation_count > 0:
            print(self.get_warning_message(continuation_count, 'Translation'))
        
        print('\n' + self.config.separator)
        print("Translation complete!")
        
        await self.save_to_srt(translation_response, output_filename, 'Translation')

class ArgumentParser:
    """Handle command line argument parsing."""
    
    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        """Create and configure argument parser."""
        parser = argparse.ArgumentParser(
            description='Transcribe audio files and translate to specified language'
        )
        
        parser.add_argument(
            'file',
            type=str,
            help='Path to the audio file to transcribe'
        )
        
        parser.add_argument(
            '--no-translate', '-n',
            dest='translate',
            action='store_false',
            help='Disable translation (translation is enabled by default)'
        )
        
        parser.add_argument(
            '--language', '-l',
            type=str,
            default='English',
            help='Target language for translation (default: English)'
        )
        
        parser.add_argument(
            '--disable-merge-sign', '-ms',
            action='store_true',
            dest='disable_merge_sign',
            help='Disable showing a merge sign in the output file in case of repeated model runs'
        )
        
        parser.set_defaults(translate=True, disable_merge_sign=False)
        return parser
    
    @staticmethod
    def validate_file_path(file_path: str) -> None:
        """Validate that the input file exists."""
        if not pathlib.Path(file_path).exists():
            print(f"Error: File '{file_path}' does not exist.")
            sys.exit(1)

async def main() -> None:
    """Main application entry point."""
    parser = ArgumentParser.create_parser()
    args = parser.parse_args()
    
    ArgumentParser.validate_file_path(args.file)
    
    # Initialize transcriber
    config = TranscriptionConfig()
    transcriber = AudioTranscriber(config, args.disable_merge_sign)
    
    # Prepare file and transcribe
    processed_file = AudioTranscriber.prepare_file(args.file)
    file_path = pathlib.Path(processed_file)
    
    messages = await transcriber.transcribe(file_path)
    
    # Handle translation
    if args.translate:
        await transcriber.translate(file_path, messages, args.language)
    else:
        print("Skipping translation due to --no-translate flag.\n\n")

if __name__ == "__main__":
    asyncio.run(main())
