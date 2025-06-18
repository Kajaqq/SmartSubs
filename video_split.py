from av import open
from pathlib import PurePath

extension_map = {
    'aac': 'aac',
    'mp3': 'mp3',
    'ac3': 'ac3',
    'opus': 'opus',
    'vorbis': 'ogg',
    'flac': 'flac',
    'pcm_s16le': 'wav'
}


def detect_video(file_path):
    with open(file_path, 'r') as container:
        if container.streams.video:
            return True
        else:
            return False
    return None

def get_audio_codec(video_path):
    with open(video_path, 'r') as container:
        if not container.streams.audio:
            raise Exception("No audio streams found")
        audio_stream = container.streams.audio[0]
    codec_context = audio_stream.codec_context
    codec_name = codec_context.name
    return codec_name

def get_output_extension(codec_name):
    return extension_map.get(codec_name, codec_name)

def get_output_path(video_path):
    audio_codec = get_audio_codec(video_path)
    output_extension = get_output_extension(audio_codec)
    output_path_object = PurePath(video_path).with_suffix(f".{output_extension}")
    output_path = str(output_path_object)
    return output_path

def extract_audio(video_path, audio_path):
    print(f"Extracting audio to '{audio_path}'...")
    try:
        input_container  = open(video_path, 'r')
        output_container = open(audio_path, 'w')
        input_stream = input_container.streams.audio[0]
        output_stream = output_container.add_stream_from_template(input_stream)
        for packet in input_container.demux(input_stream):
            if packet.dts is None:
                    continue
            packet.stream = output_stream
            output_container.mux(packet)
        input_container.close()
        output_container.close()
        return True
    except FileNotFoundError:
        print(f"Error: File '{video_path}' not found.")
        return False
    except Exception as e:
        print(f"An error occurred during extraction: {e}")
        return False
    except IndexError:
        print(f"Error: No audio stream found in '{video_path}' for extraction.")
        return False

if __name__ == "__main__":
    input_file = 'video_file.mp4'
    if not detect_video(input_file):
        print("The input file is not a video.")
        exit(1)
    input_video = input_file
    output_audio = get_output_path(input_video)
    print(f"Extracting audio from '{input_video}'...")
    extract_audio(input_video, output_audio)
    print(
        f"Audio extracted successfully to '{output_audio}'."
    )
