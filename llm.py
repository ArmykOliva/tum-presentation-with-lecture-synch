import os, json, base64, re
from time import sleep
from datetime import datetime, date
from openai import OpenAI, RateLimitError
from jinja2 import Template
from prompts import *
from traceback import print_exc
from PIL import Image
from io import BytesIO


openrouter_client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

def render_messages(messages,images=[], **kwargs):
    # Create a new list to hold the rendered prompts
    rendered_messages = []
    # Iterate over each prompt in the messages list
    for message in messages:
        template = Template(message["content"])
        # Add zip function to the context
        context = kwargs.copy()
        context['zip'] = zip
        rendered_content = template.render(**context)
        rendered_messages.append({"role": message["role"], "content": rendered_content})

    if images:
        rendered_messages[-1] = {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": rendered_messages[-1]["content"]
                }
            ]
        }
        for image_path in images:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                
                _, ext = os.path.splitext(image_path)
                ext = ext.lower()
                
                if ext in ['.png']:
                    # Convert PNG to JPEG
                    image = Image.open(BytesIO(image_data)).convert("RGB")
                    buffer = BytesIO()
                    image.save(buffer, format="JPEG")
                    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    mime_type = 'image/jpeg'
                elif ext in ['.jpg', '.jpeg']:
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                    mime_type = 'image/jpeg'
                else:
                    # Skip unsupported file types
                    continue

                rendered_messages[-1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}",
                        "detail": "auto"
                    }
                })

    # if kwarg "model" contains "gemma" then add the content of the system message to the first user message on the top and remove the system message
    if 'model' in kwargs and 'gemma' in kwargs['model'].lower():
        system_message = next((msg for msg in rendered_messages if msg['role'] == 'system'), None)
        if system_message:
            # Find the first user message
            first_user_message = next((msg for msg in rendered_messages if msg['role'] == 'user'), None)
            if first_user_message:
                # If the first user message is already a list of content (for images), prepend to the text content
                if isinstance(first_user_message['content'], list):
                    first_user_message['content'][0]['text'] = system_message['content'] + '\n\n' + first_user_message['content'][0]['text']
                else:
                    first_user_message['content'] = system_message['content'] + '\n\n' + first_user_message['content']
            
            # Remove the system message
            rendered_messages = [msg for msg in rendered_messages if msg['role'] != 'system']
            
    return rendered_messages

def call_pure_llm(rendered_messages, model="meta-llama/llama-3.1-8b-instruct",temperature=0,max_tokens=6000,json_schema=None,stream=False,client=None,**kwargs):
    used_client = openrouter_client

    if (client): used_client = client
    
    while True:
        try:
            config = {
                "model": model,
                "messages": rendered_messages,
                "temperature": temperature,
                # "max_tokens": max_tokens,
                "stream": True
            }
            if json_schema:
                # Remove single-line comments, but ignore "//" if preceded by ":"
                json_string = re.sub(r'(?<!:)//.*$', '', json_schema, flags=re.MULTILINE)
                
                # Remove multi-line comments
                json_string = re.sub(r'/\*.*?\*/', '', json_string, flags=re.DOTALL)
                print(json_string[:100])

                config["extra_body"] = {
                    "guided_decoding_backend": "lm-format-enforcer",
                    "guided_json": json_string
                }

            response = used_client.chat.completions.create(**config)

            if stream:
                return response
            else:
                response_string = ""
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if (content):
                        response_string += content
                return response_string
        except RateLimitError:
            print("Rate limited, waiting 10 seconds")
            print_exc()
            sleep(10)

def call_llm(messages_template, images=[], **kwargs) -> str:
    # Render the messages
    rendered_messages = render_messages(messages_template,images=images,**kwargs)

    # Call the LLM
    response = call_pure_llm(rendered_messages, stream=False,**kwargs)

    return response

def call_llm_stream(messages_template, images=[], **kwargs):
    rendered_messages = render_messages(messages_template, images=images, **kwargs)
    response = call_pure_llm(rendered_messages, stream=True, **kwargs)
    
    def response_generator():
        full_response = ""
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                full_response += content
                yield full_response
    return response_generator()

def call_llm_json(messages_template,**kwargs):
    # Render the messages
    rendered_messages = render_messages(messages_template,**kwargs)

    i = 0
    while True:
        try:
            # Call the LLM
            response = call_pure_llm(rendered_messages,**kwargs)
            data = json.loads(response[response.find('{'):response.rfind('}')+1])
            break
        except json.JSONDecodeError:
            print("invalid json", response)
            i += 1
            if (i > 4 or len(response) >= 30):
                raise Exception("Invalid JSON response from LLM")

    return data