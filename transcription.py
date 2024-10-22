from typing import Any, List, Optional
import datetime
import subprocess
import os
import time
import torch
from transformers import pipeline
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn

class WhisperTools():
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        model_name = "openai/whisper-large-v3"
        self.model = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            torch_dtype=torch.float16,
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_kwargs={"attn_implementation": "sdpa"},
        )

        print("cuda", torch.cuda.is_available(), torch.version.cuda)
        print("Whisper loaded")

    def transcribe(
        self,
        file: str,
        num_speakers: Optional[int] = None,
        language: Optional[str] = None,
        prompt: str = "",
    ):
        try:
            # Generate a temporary filename
            temp_wav_filename = f"temp-{time.time_ns()}.wav"

            # Convert the audio file to a WAV file
            if file is not None:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        file,
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-c:a",
                        "pcm_s16le",
                        temp_wav_filename,
                    ]
                )

            transcript, chunks = self.speech_to_text(
                temp_wav_filename,
                num_speakers,
                prompt=prompt,
                language=language,
            )

            print(f"done with inference")
            # Return the results as a JSON object
            return transcript, chunks

        except Exception as e:
            raise RuntimeError("Error Running inference with local model", e)

        finally:
            # Clean up
            if os.path.exists(temp_wav_filename):
                os.remove(temp_wav_filename)

    def convert_time(self, secs, offset_seconds=0):
        return datetime.timedelta(seconds=(round(secs) + offset_seconds))

    def speech_to_text(
        self,
        audio_file_wav,
        num_speakers=None,
        prompt="",
        offset_seconds=0,
        group_segments=True,
        language=None,
        translate=False,
    ):
        time_start = time.time()

        # Transcribe audio
        print("Starting transcribing")
        generate_kwargs = {
            "task": "translate" if translate else "transcribe",
            "language": language if language != "None" else None,
        }

        with Progress(
            TextColumn("🤗 [progress.description]{task.description}"),
            BarColumn(style="yellow1", pulse_style="white"),
            TimeElapsedColumn(),
        ) as progress:
            progress.add_task("[yellow]Transcribing...", total=None)

            outputs = self.model(
                audio_file_wav,
                chunk_length_s=30,
                batch_size=12,
                generate_kwargs=generate_kwargs,
                return_timestamps=True,
            )

        transcript = outputs["text"]
        chunks = outputs["chunks"]
        
        time_transcribing_end = time.time()
        print(
            f"Finished with transcribing, took {time_transcribing_end - time_start:.5} seconds"
        )

        time_end = time.time()
        time_diff = time_end - time_start

        system_info = f"""Processing time: {time_diff:.5} seconds"""
        print(system_info)
        return transcript, chunks