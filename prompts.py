from re import L

SLIDE_ANALYSIS_PROMPT = [
    {"role": "system", "content": """
You are an expert in analyzing presentation slides and their corresponding transcripts. Your task is to identify information mentioned in the transcript that is not present on the slide. Focus on key points, important details, or explanations that would enhance the slide's content.

Please return your analysis as a JSON list of items to add to the slide. If there's nothing significant to add, return an empty list.
Each addition can be a single sentence or a short phrase. Especially any information about Exams, dates, times, links, sources, events are important. If there is nothing important to add, or everything was already stated, return an empty list.
MAKE SURE TO NOT REPEAT ANYTHING THAT IS ALREADY ON THE SLIDE. ITS ALWAYS BETTER TO REPLY WITH AN EMPTY LIST OR LESS ITEMS THAN MORE.
The json should be formatted like this:
{
    "additions": [
        "item 1",
        "item 2",
        "item 3",
     .... more
    ]
}
write it in english.
""".strip()},
    {"role": "user", "content": """
Analyze the following slide image, its text content, and its corresponding transcript. Identify any important information from the transcript that is missing from the slide.

Slide number: {{ slide_number }}

Slide text:
'''
{{ slide_text }}
'''

Transcript:
'''
{{ transcript }}
'''

Provide your response as a JSON list of strings, each representing an item to add to the slide. If there's nothing significant to add, return an empty list. write it in english.
""".strip()}
]
