from PIL import Image
import os
import fitz
from torchvision import transforms
from transformers import AutoModelForObjectDetection
import torch
import shutil
import re


"""

This file is able of transforming a whole PDF into images and then use 2 models to detect
and crop the tables from each page. It is quite fast but not as accurate as we would like maybe. 

"""

class TableExtractor:
    class MaxResize(object):
        def __init__(self, max_size):
            self.max_size = max_size

        def __call__(self, image):
            width, height = image.size
            scale = self.max_size / max(width, height)
            return image.resize((int(scale * width), int(scale * height)))

    def __init__(self, pdf_path, pages_folder, cropped_table_directory, max_detection_size=800, max_structure_size=1000):
        self.pdf_path = pdf_path
        self.pages_folder = pages_folder
        self.cropped_table_directory = cropped_table_directory
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        shutil.rmtree(self.pages_folder, ignore_errors=True)
        os.makedirs(self.pages_folder, exist_ok=True)

        shutil.rmtree(self.cropped_table_directory, ignore_errors=True)
        os.makedirs(self.cropped_table_directory, exist_ok=True)

        self.detection_transform = transforms.Compose([
            TableExtractor.MaxResize(max_detection_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.structure_transform = transforms.Compose([
            TableExtractor.MaxResize(max_structure_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        print('Start loading table models')
        self.model = AutoModelForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection", revision="no_timm"
        ).to(self.device)

        self.structure_model = AutoModelForObjectDetection.from_pretrained(
            "microsoft/table-transformer-structure-recognition-v1.1-all"
        ).to(self.device)

        print('Table models loaded')

    
            
    def convert_pdf_to_images(self,max_pages):
        pdf_document = fitz.open(self.pdf_path)
        for page_number in range(0,max_pages):
            page = pdf_document[page_number]
            pix = page.get_pixmap()
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            image.save(os.path.join(self.pages_folder, f"page_{page_number + 1}.png"))
            
        pdf_document.close()


    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(self,x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)


    def rescale_bboxes(self,out_bbox, size):
        width, height = size
        boxes = self.box_cxcywh_to_xyxy(out_bbox)
        boxes = boxes * torch.tensor(
            [width, height, width, height], dtype=torch.float32
        )
        return boxes


    def outputs_to_objects(self,outputs, img_size, id2label):
        m = outputs.logits.softmax(-1).max(-1)
        pred_labels = list(m.indices.detach().cpu().numpy())[0]
        pred_scores = list(m.values.detach().cpu().numpy())[0]
        pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]
        pred_bboxes = [
            elem.tolist() for elem in self.rescale_bboxes(pred_bboxes, img_size)
        ]

        objects = []
        for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
            class_label = id2label[int(label)]
            if not class_label == "no object":
                objects.append(
                    {
                        "label": class_label,
                        "score": float(score),
                        "bbox": [float(elem) for elem in bbox],
                    }
                )

        return objects


    def detect_tables_in_cropped_image(self,cropped_image):
        """
        Detect tables in the cropped image using the model.
        """
        pixel_values = self.detection_transform(cropped_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(pixel_values)

        detected_tables = self.outputs_to_objects(outputs, cropped_image.size, self.model.config.id2label)
        return len(detected_tables) > 0  # Returns True if any table is detected

    def detect_and_crop_save_table(self,file_path, cropped_table_directory):
        image = Image.open(file_path)
        filename, _ = os.path.splitext(file_path.split("/")[-1])

        pixel_values = self.detection_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(pixel_values)

        id2label = self.model.config.id2label
        id2label[len(self.model.config.id2label)] = "no object"
        detected_tables = self.outputs_to_objects(outputs, image.size, id2label)
        for idx, table in enumerate(detected_tables):
            cropped_table = image.crop(table["bbox"])
            cropped_table_path = f"./{cropped_table_directory}/{filename}_{idx}.png"
            cropped_table.save(cropped_table_path)

            # Reapply detection on the cropped table
            if not self.detect_tables_in_cropped_image(cropped_table):
                print(f"No table detected in cropped image: {cropped_table_path}")
                os.remove(cropped_table_path)

    def process_all_pages(self, max_pages):
            print('Converting to pdf images')
            self.convert_pdf_to_images(max_pages)
            print('Converted')
            files_with_pages = []
            for file_name in os.listdir(self.pages_folder):
                # Extract page number from file name using regular expression
                match = re.search(r"page_(\d+).png", file_name)
                if match:
                    page_number = int(match.group(1))
                    files_with_pages.append((file_name, page_number))

            # Sort the list based on page numbers
            files_with_pages.sort(key=lambda x: x[1])

            # Iterate through the sorted file list
            for file_name, _ in files_with_pages[:max_pages]:
                full_path = os.path.join(self.pages_folder, file_name)
                self.detect_and_crop_save_table(full_path, self.cropped_table_directory)
