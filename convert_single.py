import argparse
from marker.convert import convert_single_pdf
from marker.logger import configure_logging
from marker.models import load_all_models
import json
import os
import time
from pypdf import PdfReader
import re
import shutil
import PyPDF2
from concurrent.futures import ProcessPoolExecutor, wait
from extract_tables import TableExtractor
from create_image_from_tables import ImageComposer

configure_logging()
"""
Extraer solo imágenes y tablas hasta max_pages.
"""

### PYPDF2 o "unstructured.io" la versión "fast" - > PDFMiner. ### Esto tarda poquísimo. 

def extract_pypdf(fname):
    start = time.time()

    with open(fname, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        pages_text = []
        for page_num in range(len(reader.pages)):
            page_info = {
                "page_number": page_num + 1,
                "text": reader.pages[page_num].extract_text()
            }
            pages_text.append(page_info)
    end = time.time()
    print(f'It took {str(end-start)} to extract using pypdf.')
    return pages_text


### Modelos deberían estar precargados de alguna manera. 
def calculate_markdown(fname, args):
    start = time.time()
    model_lst = load_all_models()
    end = time.time()
    print(f'It took {str(end-start)} seconds to load the models.')
    print('Started converting to markdown.')
    start = time.time()
    full_text, out_meta = convert_single_pdf(fname, model_lst, max_pages=args.max_pages, parallel_factor=args.parallel_factor)
    end = time.time()
    print(f'Finished converting to markdown. It lasted {str(end-start)}')
    with open(args.output, "w+", encoding='utf-8') as f:
        f.write(full_text)

    out_meta_filename = args.output.rsplit(".", 1)[0] + "_meta.json"
    with open(out_meta_filename, "w+") as f:
        f.write(json.dumps(out_meta, indent=4))

def read_and_save_images(args):
    start = time.time()

    shutil.rmtree(args.images_path, ignore_errors=True)
    os.makedirs(args.images_path, exist_ok=True)

    reader = PdfReader(args.filename)
    for page_num, page in enumerate(reader.pages):
        if page_num > args.max_pages:
            break
        for index, image in enumerate(page.images):
            image_name = image.name
            image_name = image_name.split('.')
            image_name = f'{image_name[0]}_{index}_{page_num+1}.{image_name[1]}'

            with open(os.path.join(args.images_path, image_name), "wb") as fp:
                fp.write(image.data)
    end = time.time()
    print(f'Images extracted. It took {str(end-start)}')
    
def read_and_save_tables(args, tables_togpt4_path):
    pdf_path = args.filename
    pages_folder = args.pages_folder
    cropped_tables_directory = args.cropped_tables_directory

    detector = TableExtractor(pdf_path, pages_folder, cropped_tables_directory)
    detector.process_all_pages(args.max_pages)
    print('Detected all the tables')

    composer = ImageComposer(cropped_tables_directory, pdf_path)
    composer.compose_images(tables_togpt4_path)

def gpt4_tables():
    pass

def postprocess_markdown(args, pages_text):
    # Leer el contenido del archivo Markdown
    with open(args.output, 'r', encoding='utf-8') as file:
        content = file.read()

    # Preparar para escribir el contenido modificado
    new_content = []

    # Expresión regular para encontrar las etiquetas de página
    page_tag_pattern = r"\[comment-marker\]: # \(page (\d+) start\)"
    
    # Dividir el contenido en secciones basadas en las etiquetas de página
    markdown_pages = re.split(page_tag_pattern, content)

    # Crear un diccionario para mapear el texto extraído de pypdf por número de página
    pypdf_pages = {page['page_number']: page['text'] for page in pages_text}

    image_filename_pattern = re.compile(r"_([0-9]+)\.(jpg|png)$")

    # Reensamblar el contenido, insertando el texto de pypdf donde sea necesario
    for i in range(1, len(markdown_pages), 2):
        page_number = int(markdown_pages[i])
        page_content = markdown_pages[i+1]

        new_content.append(f"[comment-marker]: # (page {page_number} start)\n")  # Reinsertar la etiqueta de página
        new_content.append(page_content)  # Añadir el contenido de la página de markdown

        # Obtener el texto correspondiente de PyPDF2
        pypdf_text = pypdf_pages.get(page_number, '')

        # Si el texto de PyPDF2 cumple con la condición, añadir como comentario
        if pypdf_text and len(pypdf_text.strip()) >= 2 * len(page_content.strip()):
            comment_text = f"<!--\n{pypdf_text}\n-->\n"
            new_content.append(comment_text)

        for img_file in sorted(os.listdir(args.images_path)):
            match = image_filename_pattern.search(img_file)
            if match:
                img_page_number = int(match.group(1))
                if img_page_number == page_number:
                    image_path = os.path.join(args.images_path, img_file)
                    image_markup = f'![image](./{image_path})\n'
                    new_content.append(image_markup)

        # Aquí añadirías las imágenes o cualquier otro contenido necesario
        

    # Escribir el contenido modificado en un nuevo archivo Markdown
    with open('with_images_' + args.output, 'w', encoding='utf-8') as file:
        file.writelines(new_content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="PDF file to parse")
    parser.add_argument("output", help="Output file name")
    parser.add_argument("--max_pages", type=int, default=None, help="Maximum number of pages to parse")
    parser.add_argument("--parallel_factor", type=int, default=1, help="How much to multiply default parallel OCR workers and model batch sizes by.")
    parser.add_argument("--images_path", type=str, help = "Where to save images")
    parser.add_argument("--pages_folder", type=str, help = "Where to save the screenshots of the PDF. ")
    parser.add_argument("--cropped_tables_directory", type=str, help = "Where to save extracted tables. ")
    args = parser.parse_args()

    tables_to_gpt4_path = f'{args.cropped_tables_directory}_togpt4'

    # Crear y empezar los hilos
    with ProcessPoolExecutor(max_workers=4) as executor:
        text_pypdf = executor.submit(extract_pypdf, args.filename)
        future_markdown = executor.submit(calculate_markdown, args.filename, args)
        future_imagenes = executor.submit(read_and_save_images, args)
        future_tables = executor.submit(read_and_save_tables, args, tables_to_gpt4_path)

        textpypdf = text_pypdf.result()
        wait([future_imagenes, future_markdown, future_tables]) 
        
    postprocess_markdown(args,textpypdf)


if __name__ == "__main__":
    main()
