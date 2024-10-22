# this is the most disguisting piece of shit code i have ever written, i am sorry for everyone involved
# please forgive me
# but it works so fucking well
# this is the most over engineered AI solution that you would never find anywhere else, its actually genious
# I am almost 100% sure that if you made it faster and bundeled into a webapp you could make a lot of fucking money
# If you do that pls buy me a beer
import tempfile
import torch
import json
import yaml
import subprocess
import os
import cv2
import numpy as np
import easyocr
from pdf2image import convert_from_path
from moviepy.editor import VideoFileClip
from fuzzywuzzy import fuzz
from tqdm import tqdm
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
from llm import call_llm_json
from prompts import SLIDE_ANALYSIS_PROMPT
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph

from transcription import WhisperTools

VIDEO_CHUNK = 10

# Load configuration from config.yaml
with open('config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Ensure the videos directory exists
os.makedirs('videos', exist_ok=True)

# Initialize EasyOCR reader
reader = easyocr.Reader(['de'], gpu=True)

# Function to download a video
def download_video(lecture_name, url, index):
    output_filename = f"videos/{lecture_name}_{url[-10:]}.mp4"
    
    if os.path.exists(output_filename):
        print(f"File '{output_filename}' already exists. Skipping download.")
        return
    
    command = [
        'ffmpeg',
        '-protocol_whitelist', 'file,http,https,tcp,tls',
        '-i', url,
        '-bsf:a', 'aac_adtstoasc',
        '-vcodec', 'copy',
        '-c', 'copy',
        '-crf', '50',
        '-preset', 'fast',
        '-loglevel', 'quiet',
        output_filename
    ]
    
    print(f"Downloading {lecture_name} (Part {index + 1})...")
    subprocess.run(command)
    print(f"Download completed. File saved as '{output_filename}'")

# Process each lecture in the config
for lecture_name, urls in config['lectures'].items():
    for index, url in enumerate(urls):
        download_video(lecture_name, url, index)

print("All downloads completed.")

def extract_text_from_image(image):
    # Convert PIL Image to numpy array
    if isinstance(image, np.ndarray):
        # If it's already a numpy array, use it as is
        img_array = image
    else:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
    
    result = reader.readtext(img_array)
    text = ' '.join([text for _, text, _ in result])
    return text

def compare_texts(text1, text2):
    return fuzz.ratio(text1, text2)

def find_matching_slide(video_frame_text, slide_texts):
    best_match = None
    best_score = 0
    
    for i, slide_text in enumerate(slide_texts):
        score = compare_texts(video_frame_text, slide_text)
        if score > best_score:
            best_score = score
            best_match = i
    
    return best_match if best_score > 80 else None

def process_presentation(slide_texts, video_path):
    # Load video
    print("Loading video...")
    video = VideoFileClip(video_path)
    total_duration = int(video.duration)
    print(f"Loaded video. Duration: {total_duration} seconds")
    
    # Process video frames
    print("Processing video frames...")
    results = []
    for t in tqdm(range(0, total_duration, VIDEO_CHUNK)):
        frame = video.get_frame(t)
        frame_text = extract_text_from_image(frame)
        print(f"Frame {t}s of {total_duration}s")
        
        if frame_text.strip():  # Check if there's any text in the frame
            matched_slide = find_matching_slide(frame_text, slide_texts)
            if matched_slide is not None:
                results.append((t, matched_slide))
                print(f"  Slide match found at {t}s. slide: {matched_slide}")
    
    # Remove duplicate slide changes
    print("Removing duplicate slide changes...")
    final_timestamps = []
    current_slide = None
    for t, slide in results:
        if slide != current_slide:
            final_timestamps.append({'timestamp': t, 'slide': slide})
            current_slide = slide
            print(f"Unique slide match found at {t}s. Slide: {slide}")
    
    print(f"Finished processing. Found {len(final_timestamps)} unique slide matches.")
    return final_timestamps
# Process each presentation in the presentations folder
presentations_folder = 'presentations'

# Load existing results if available
try:
    with open('slide_timestamps.json', 'r', encoding='utf-8') as file:
        results = json.load(file) or {}
except FileNotFoundError:
    results = {}
    # make the file with empty dict
    with open('slide_timestamps.json', 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, indent=2)

already_done_videos = set()
for filename in os.listdir(presentations_folder):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(presentations_folder, filename)
        lecture_name = os.path.splitext(filename)[0]
        
        # Check if lecture_name is already processed
        if lecture_name in results:
            print(f"Skipping lecture: {lecture_name} (already processed)")
            continue
        
        print(f"Processing lecture: {lecture_name}")

        print(f"Converting PDF to images...")
        slides = convert_from_path(pdf_path)
        print(f"Converted {len(slides)} slides to images.")
        
        print(f"Extracting text from slides...")
        slide_texts = []
        for i, slide in enumerate(tqdm(slides, desc="Extracting text", unit="slide")):
            slide_text = extract_text_from_image(slide)
            slide_texts.append(slide_text)
            if (i + 1) % 5 == 0:
                print(f"Processed {i + 1}/{len(slides)} slides")
        print(f"Finished extracting text from all {len(slides)} slides.")
            
        lecture_results = []
        video_paths = os.listdir('videos')
        
        for i, video_name in enumerate(video_paths):
            print(f"Processing video: {video_name}")
            video_path = os.path.join('videos', video_name)
            
            if os.path.exists(video_path):
                slide_timestamps = process_presentation(slide_texts, video_path)
                if (len(slide_timestamps) > 3 and video_name not in already_done_videos):
                    already_done_videos.add(video_name)
                    lecture_results.append({
                        'video': {
                            'name': video_name,
                            'timestamps': slide_timestamps,
                            'transcript': None
                        }
                    })
        
        results[lecture_name] = lecture_results
        
        # Save results after processing each lecture
        with open('slide_timestamps.json', 'w', encoding='utf-8') as outfile:
            json.dump(results, outfile, indent=2)
        
        print(f"Results for {lecture_name} saved in slide_timestamps.json")

print("Processing completed. All results saved in slide_timestamps.json")

del reader
# unload the ocr model from vram
torch.cuda.empty_cache()    

# speech to text using transcription.py
print("Setting up Whisper Tools...")
whisper_tools = WhisperTools()
whisper_tools.setup()
print("Whisper Tools setup complete")

# Load existing slide_timestamps.json
with open('slide_timestamps.json', 'r', encoding='utf-8') as file:
    slide_timestamps = json.load(file)

# Process each presentation in slide_timestamps.json
for lecture_name, lecture_data in slide_timestamps.items():
    print(f"Processing lecture: {lecture_name}")
    
    for video_data in lecture_data:
        video_name = video_data['video']['name']
        video_path = os.path.join('videos', video_name)
        
        # Skip if transcript already exists
        if "transcript" in video_data['video'] and video_data['video']['transcript'] is not None:
            print(f"Skipping transcription for {video_name} (already processed)")
            continue
        
        print(f"Processing video: {video_name}")
        
        # Generate transcript using WhisperTools
        print(f"Generating transcript for {video_name}")
        transcript, chunks = whisper_tools.transcribe(video_path)
        
        # Update video data with chunks instead of full transcript
        video_data['video']['transcript'] = chunks
    
    # Save updated results after processing each lecture
    with open('slide_timestamps.json', 'w', encoding='utf-8') as outfile:
        json.dump(slide_timestamps, outfile, indent=2)
    
    print(f"Updated results for {lecture_name} saved in slide_timestamps.json")

print("Transcription process completed. All results saved in slide_timestamps.json")

# Unload Whisper model from VRAM
del whisper_tools
torch.cuda.empty_cache()

print("Processing completed.")

# for each slide map the chunks of transcript

def map_transcript_to_slides(timestamps, transcript):
    slide_transcripts = {}
    current_slide = timestamps[0]['slide']
    current_transcript = []
    
    for chunk in transcript:
        chunk_start = chunk['timestamp'][0]
        
        # Find the correct slide for this chunk
        for i, timestamp in enumerate(timestamps):
            if i + 1 < len(timestamps) and chunk_start >= timestamp['timestamp'] and chunk_start < timestamps[i+1]['timestamp']:
                if timestamp['slide'] != current_slide:
                    slide_transcripts[current_slide] = ' '.join(current_transcript)
                    current_slide = timestamp['slide']
                    current_transcript = []
                break
        
        current_transcript.append(chunk['text'])
    
    # Add the last slide's transcript
    slide_transcripts[current_slide] = ' '.join(current_transcript)
    
    return slide_transcripts

# Load existing slide_timestamps.json
with open('slide_timestamps.json', 'r', encoding='utf-8') as file:
    slide_timestamps = json.load(file)

# Process each presentation in slide_timestamps.json
for lecture_name, lecture_data in slide_timestamps.items():
    print(f"Processing lecture: {lecture_name}")
    
    for video_data in lecture_data:
        video_name = video_data['video']['name']
        
        # Skip if slide_voice_transcript already exists
        if "slide_voice_transcript" in video_data['video']:
            print(f"Skipping slide voice transcript for {video_name} (already processed)")
            continue
        
        print(f"Processing video: {video_name}")
        
        # Map transcript to slides
        slide_transcripts = map_transcript_to_slides(video_data['video']['timestamps'], video_data['video']['transcript'])
        
        # Add slide_voice_transcript to video data
        video_data['video']['slide_voice_transcript'] = slide_transcripts
    
    # Save updated results after processing each lecture
    with open('slide_timestamps.json', 'w', encoding='utf-8') as outfile:
        json.dump(slide_timestamps, outfile, indent=2)
    
    print(f"Updated results for {lecture_name} saved in slide_timestamps.json")

print("Slide voice transcript mapping completed. All results saved in slide_timestamps.json")

# load slide_timestamps.json and print it
with open('slide_timestamps.json', 'r', encoding='utf-8') as file:
    slide_timestamps = json.load(file)

lecture_slides = {}
for lecture_name, lecture_data in slide_timestamps.items():
    print(f"Lecture: {lecture_name}")
    lecture_slide_voice_transcripts = {}
    for video_data in lecture_data:
        print(f"  Video: {video_data['video']['name']}")
        for slide in video_data['video']['slide_voice_transcript']:
            if (slide not in lecture_slide_voice_transcripts):
                lecture_slide_voice_transcripts[slide] = ""
            lecture_slide_voice_transcripts[slide] += f" -- {video_data['video']['slide_voice_transcript'][slide]} "
    lecture_slides[lecture_name] = lecture_slide_voice_transcripts

#do the prompting
def create_text_pdf(text, filename):
    packet = BytesIO()
    doc = SimpleDocTemplate(packet, pagesize=letter, topMargin=72, bottomMargin=72, leftMargin=72, rightMargin=72)
    
    styles = getSampleStyleSheet()
    style = styles['Normal']
    style.fontSize = 12
    style.leading = 14
    
    flowables = []
    paragraphs = text.split('\n\n')
    for para in paragraphs:
        p = Paragraph(para, style)
        flowables.append(p)
    
    doc.build(flowables)
    
    packet.seek(0)
    return PdfReader(packet)

def extract_text_from_pdf_page(pdf_reader, page_number):
    return pdf_reader.pages[page_number].extract_text()

# Create the edited_presentations folder if it doesn't exist
edited_presentations_folder = 'edited_presentations'
os.makedirs(edited_presentations_folder, exist_ok=True)

# Assuming 'presentations' is the folder containing the original PDFs
presentations_folder = 'presentations'

# Load existing slide_timestamps.json
with open('slide_timestamps.json', 'r', encoding='utf-8') as file:
    slide_timestamps = json.load(file)

for lecture_name, lecture_slides in lecture_slides.items():
    output_pdf_path = os.path.join(edited_presentations_folder, f"{lecture_name}_updated.pdf")
    
    # Check if the edited presentation already exists
    if os.path.exists(output_pdf_path):
        print(f"Skipping lecture: {lecture_name} (edited presentation already exists)")
        continue
    
    print(f"LLM Processing lecture: {lecture_name}")
    original_pdf_path = os.path.join(presentations_folder, f"{lecture_name}.pdf")
    
    original_pdf = PdfReader(original_pdf_path)
    pdf_writer = PdfWriter()
    
    # Convert PDF pages to images
    pdf_images = convert_from_path(original_pdf_path)
    
    # Ensure lecture exists in slide_timestamps
    if lecture_name not in slide_timestamps:
        slide_timestamps[lecture_name] = []

    # Ensure 'llm_outputs' exists for this lecture
    for video_data in slide_timestamps[lecture_name]:
        if 'llm_outputs' not in video_data['video']:
            video_data['video']['llm_outputs'] = {}
    
    for slide_number in range(len(original_pdf.pages)):
        # Add the original slide
        pdf_writer.add_page(original_pdf.pages[slide_number])
        
        # Check if this slide has additional information
        if str(slide_number) in lecture_slides:
            transcript = lecture_slides[str(slide_number)]

            # Extract text from the current slide
            slide_text = extract_text_from_pdf_page(original_pdf, slide_number)
            
            # Check if LLM output already exists for this slide
            llm_output = None
            for video_data in slide_timestamps[lecture_name]:
                if str(slide_number) in video_data['video']['llm_outputs']:
                    llm_output = video_data['video']['llm_outputs'][str(slide_number)]
                    print(f"Using existing LLM output for slide {slide_number}")
                    break
            
            if llm_output is None:
                # Save the current slide image to a temporary file
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_image:
                    pdf_images[slide_number].save(temp_image.name, 'PNG')
                    temp_image_path = temp_image.name
                
                # Analyze the slide and transcript
                llm_output = call_llm_json(
                    SLIDE_ANALYSIS_PROMPT,
                    slide_number=slide_number,
                    slide_text=slide_text,
                    transcript=transcript,
                    images=[temp_image_path],
                    model="anthropic/claude-3.5-sonnet"
                )
                print(f"LLM response for slide {slide_number}: {llm_output}")

                # Remove the temporary image file
                os.unlink(temp_image_path)
                
                # Save the LLM output to slide_timestamps.json
                for video_data in slide_timestamps[lecture_name]:
                    video_data['video']['llm_outputs'][str(slide_number)] = llm_output
                
                # Save updated slide_timestamps.json
                with open('slide_timestamps.json', 'w', encoding='utf-8') as outfile:
                    json.dump(slide_timestamps, outfile, indent=2)
            
            # If there are items to add, create a new slide with the additions
            if llm_output["additions"]:
                additions_text = "Additions to previous slide:\n\n" + "\n\n".join(f"- {item}" for item in llm_output["additions"])
                new_slide = create_text_pdf(additions_text, f"slide_{slide_number}_additions")
                pdf_writer.add_page(new_slide.pages[0])
        else:
            print(f"No additional information for slide {slide_number}")
    
    # Save the updated PDF in the edited_presentations folder
    with open(output_pdf_path, "wb") as output_file:
        pdf_writer.write(output_file)
    
    print(f"Updated PDF saved as: {output_pdf_path}")

print("All presentations have been processed and updated.")