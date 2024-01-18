import base64
import json
import re
import requests
import os

MODEL = "gpt-4-vision-preview"
TEMPERATURE = 1
SEED = 42
MAX_TOKENS = 4096
URL = "https://api.openai.com/v1/chat/completions"
API_KEY = ""

SYSTEM_PROMPT  = """You are a helpful table to Markdown converter, with a lot of experience in tables. I need your help with the next problem.
This image with grey background contains tables.
Those tables have their title in a yellow box with red text just above them.
Take this special considerations about the tables:
- Some of the initial columns of some tables have no header, take special care about that, because we need all the columns.
- Some of the tables are not tables, so also, take special care about that.
I want them converted into Markdown, encapsulated in a JSON array of objects. Each table will be a JSON object with the following properties:
- title: The title of the table, as is, keeping the case.
- isTable: A boolean that says if the table is really a table or not.
- content: The table content transformed to Markdown.
- summary: A summary or explanation of the table as Markdown.
As extra, if the table was not a table, I want its properties content and summary to be null.
As output, I only want the JSON, no comments about anything else.
If you do a nice job, I'll give you a generous tip."""

class OPENAI_VISION:
    def __init__(self, tables_togpt4_path):
        self.tables_togpt4_path = tables_togpt4_path

    def call_openAI(self,base64_image):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        body = {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "messages": messages
        }
        response = requests.post(URL, headers=headers, json=body)
        response = response.json()

    def execute(self):

        for image in os.listdir(self.tables_togpt4_path):
            image_path = os.path.join(self.tables_togpt4_path, image)
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                response = self.call_openAI(base64_image)



