import argparse

from marker.convert import convert_single_pdf
from marker.logger import configure_logging
from marker.models import load_all_models
import json
import os
import time
from pypdf import PdfReader
import re

configure_logging()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="PDF file to parse")
    parser.add_argument("output", help="Output file name")
    parser.add_argument("--max_pages", type=int, default=None, help="Maximum number of pages to parse")
    parser.add_argument("--parallel_factor", type=int, default=1, help="How much to multiply default parallel OCR workers and model batch sizes by.")
    parser.add_argument("--images_path", type=str, help = "Where to store images ")
    args = parser.parse_args()
    
    start = time.time()

    fname = args.filename
    model_lst = load_all_models()
    full_text, out_meta = convert_single_pdf(fname, model_lst, max_pages=args.max_pages, parallel_factor=args.parallel_factor)

    with open(args.output, "w+", encoding='utf-8') as f:
        f.write(full_text)

    out_meta_filename = args.output.rsplit(".", 1)[0] + "_meta.json"
    with open(out_meta_filename, "w+") as f:
        f.write(json.dumps(out_meta, indent=4))

    if not os.path.exists(args.images_path):
            os.makedirs(args.images_path)

    reader = PdfReader(args.filename)
    for page_num, page in enumerate(reader.pages):
        for index, image in enumerate(page.images):
            image_name = image.name
            image_name = image_name.split('.')
            image_name = f'{image_name[0]}_{index}_{page_num}.{image_name[1]}'
            print(image_name)
            with open(os.path.join(args.images_path, image_name), "wb") as fp:
                fp.write(image.data)
    

    # Leer el contenido del archivo Markdown
    with open(args.output, 'r', encoding='utf-8') as file:
        content = file.readlines()

    # Preparar para escribir el contenido modificado
    new_content = []

    # Expresión regular para encontrar las etiquetas de página
    page_tag_pattern = r"\[comment\]: # \(Page (\d+) Start\)"

    # Iterar sobre cada línea del contenido
    for line in content:
        new_content.append(line)
        # Buscar si la línea contiene la etiqueta de página
        match = re.search(page_tag_pattern, line)
        if match:
            page_number = match.group(1)
            # Buscar imágenes que coincidan con el número de página
            for img_file in os.listdir(args.images_path):
                patron = r"_([0-9]+)\.jpg$"
                resultado = re.search(patron, img_file)

                patron = r"_([0-9]+)\.png$"
                resultado2 = re.search(patron, img_file)
                if resultado: 
                    page_num = resultado.group(1)
                elif resultado2:
                    page_num = resultado2.group(1)
                else:
                    page_num = 'No se encontró'
                if page_number == page_num:
                    image_path = os.path.join(args.images_path, img_file)
                    # Añadir el path de la imagen justo después de la etiqueta
                    new_content.append(image_path + "\n")

    # Escribir el contenido modificado en un nuevo archivo Markdown
    with open('with_images' + args.output, 'w', encoding='utf-8') as file:
        file.writelines(new_content)

    end = time.time()
    print('It has taken:', str(end-start), 'seconds')
if __name__ == "__main__":
    main()
