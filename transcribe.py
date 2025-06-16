import asyncio
import pathlib
import sys

from dotenv import load_dotenv
from google import genai
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
########################################################
#
model_name = 'gemini-2.5-flash-preview-05-20'

system_message = '''Input: Podcast with Japanese language and multiple speakers.
Podcast topic: The podcast topic is a mobile game called 学園アイドルマスター (Gakuen Idolmaster)
Output Format: .srt file with speaker names
Guidelines: Analyze the audio file, detect the speaker names, and transcribe. If a speaker name cannot be detected, use numbers instead. Follow industry standards for subtitle length. Create accurate timestamps.'''
#########################################################
load_dotenv()

model = init_chat_model(model=model_name, model_provider="google_genai", temperature=0, audio_timestamp=True)

workflow = StateGraph(state_schema=MessagesState)


async def call_model(state: MessagesState):
    full_response = ""
    print("AI: ", end="", flush=True)

    async for chunk in model.astream(state["messages"]):
        if hasattr(chunk, 'content') and chunk.content:
            print(chunk.content, end="", flush=True)
            full_response += chunk.content

    print()

    from langchain_core.messages import AIMessage
    response_message = AIMessage(content=full_response)
    return {"messages": [response_message]}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
config = RunnableConfig(configurable={"thread_id": "gakumasu-club"})


async def upload_audio(audio_file):
    client = genai.Client()
    print("Uploading audio file...")
    myfile = await client.aio.files.upload(file=audio_file)
    file_name = myfile.name
    myfile_data = await client.aio.files.get(name=file_name)
    audio_uri = myfile_data.uri
    mime_type = myfile_data.mime_type
    audio_message = HumanMessage(content=[{"type": "media", "file_uri": audio_uri, "mime_type": mime_type}])
    return audio_message


def save_to_srt(transcript_data, filename="transcript"):
    filename = filename + '.srt'
    with open(filename, "w", encoding='utf-8') as f:
        if isinstance(transcript_data, str):
            f.write(transcript_data)
        else:
            for line in transcript_data:
                f.write(line)
    print(f"Transcript saved to {filename}")


async def transcribe(audio_file, translate, language):
    filename = pathlib.Path(audio_file).stem
    lang_suffix = language.lower().replace(' ', '_')
    try:
        audio_msg = await upload_audio(audio_file)
        messages = [
            SystemMessage(content=system_message),
            audio_msg
        ]

        print('Starting transcription...')
        print("AI Response: ", end="", flush=True)

        full_response = ""
        chunk_count = 0

        async for chunk in model.astream(messages):
            if hasattr(chunk, 'content') and chunk.content:
                print(chunk.content, end="", flush=True)
                full_response += chunk.content
                chunk_count += 1

                # Add periodic progress indicator
                if chunk_count % 50 == 0:
                    print(" ⏳", end="", flush=True)

        print(f"\nTranscription complete! ({chunk_count} chunks received)")
        save_to_srt(full_response, filename)

        if translate:
            print("\nStarting translation...")
            query2 = f"Translate the above content to {language}, output a .srt file with speaker names."
            messages.append(HumanMessage(content=query2))

            print("Translation: ", end="", flush=True)
            translation_response = ""
            chunk_count = 0

            async for chunk in model.astream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    print(chunk.content, end="", flush=True)
                    translation_response += chunk.content
                    chunk_count += 1

                    if chunk_count % 30 == 0:
                        print(" ⏳", end="", flush=True)

            print(f"\nTranslation complete! ({chunk_count} chunks received)")
            save_to_srt(translation_response, f"{filename}_{lang_suffix}")
        else:
            print("\nTranslation skipped")

    except Exception as e:
        print(f"Error during transcription: {e}")


if __name__ == "__main__":
    import argparse

    # Create argument parser
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
