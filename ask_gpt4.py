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
API_KEY = "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"

SYSTEM_PROMPT  = """You are a helpful table to Markdown converter, with a lot of experience in tables. I need your help with the next problem.
This image with grey background may contain tables.
Those tables have their title in a yellow box with red text just above them.
Take this special considerations about the tables:
- Some of the initial columns of some tables have no header, take special care about that, because we need all the columns.
- Some of the tables are not tables, so also, take special care about that.
I want them converted into Markdown, encapsulated in a JSON array of objects. Each table will be a JSON object with the following properties:
- isTable: A boolean that says if the table is really a table or not.
- content: The table content transformed to Markdown.
- caption: A detailed caption or explanation of the content of the table. 
As extra, if the table was not a table, I want its properties content and summary to be null.
As output, I only want the JSON, no comments about anything else.
If you do a nice job, I'll give you a generous tip."""

class OPENAI_VISION:
    def __init__(self, tables_togpt4_path, cropped_tables_path):
        self.tables_togpt4_path = tables_togpt4_path
        self.cropped_tables_path = cropped_tables_path

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
        return response


    def execute(self):
        with open(os.path.join(self.tables_togpt4_path, "metadata.json"), 'r') as metadata_file:
            metadata_titles = json.load(metadata_file)

        all_results = {}
        titles_to_delete = []
        for image in os.listdir(self.tables_togpt4_path):
            if not image.lower().endswith('.png'):
                continue
            image_path = os.path.join(self.tables_togpt4_path, image)
            print(image_path)

            table_titles = metadata_titles.get(image, [])
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                response = self.call_openAI(base64_image)
                print(response)
                try: 
                    response = response["choices"][0]["message"]["content"]
                    print('response 1',response)
                    response = response.strip("`").lstrip("json")
                    
                    response = re.sub(r"\s+", " ", response)
                    tables = json.loads(response)

                    for i, table in enumerate(tables):
                        if i < len(table_titles):
                            table["title"] = table_titles[i]
                    
                    all_results[image] = tables

                    data_to_write = {
                        "tables": tables
                    }
                    base_name = os.path.splitext(image)[0]
                    json_file_name = f"{base_name}_result.json"
                    json_path = os.path.join(self.tables_togpt4_path, json_file_name)
                    # Escribir en el archivo JSON
                    with open(json_path, 'w') as json_file:
                        json.dump(data_to_write, json_file, indent=4)
                    
                except Exception as e:
                    all_results[image] = {"Error": 'ERROR'}
                    print(f'Response has exception {e}')
        results_json_path = os.path.join(self.tables_togpt4_path, "all_results.json")

        for image, tables in all_results.items():
            # Asegúrate de que 'tables' sea una lista de diccionarios
            if not isinstance(tables, list):
                continue
            for table in tables:
                # Asegúrate de que 'table' sea un diccionario antes de usar 'get'
                if isinstance(table, dict) and not table.get("isTable", True):
                    title = table.get("title", "")
                    if title:
                        titles_to_delete.append(title)
                        # Construir el nombre del archivo de imagen a eliminar
                        image_to_delete = f"{title}.png"
                        image_to_delete_path = os.path.join(self.cropped_tables_path, image_to_delete)

                        # Verificar si el archivo existe y eliminarlo
                        if os.path.exists(image_to_delete_path):
                            os.remove(image_to_delete_path)
                            print(f"Imagen eliminada: {image_to_delete}")

        for image, tables in list(all_results.items()):
            all_results[image] = [table for table in tables if isinstance(table, dict) and table.get("title", "") not in titles_to_delete]
            if not all_results[image]:  # Si no hay tablas válidas, elimina la entrada de la imagen
                del all_results[image]

        with open(results_json_path, 'w') as json_file:
            json.dump(all_results, json_file, indent=4)

        


