from PIL import Image
import os
import fitz
from torchvision import transforms
from transformers import AutoModelForObjectDetection
import torch
import argparse
import shutil
from transformers import TableTransformerModel, TableTransformerConfig


"""

This file is able of transforming a whole PDF into images and then use 2 models to detect
and crop the tables from each page. It is quite fast but not as accurate as we would like maybe. 

"""
parser = argparse.ArgumentParser()
parser.add_argument("--pdf_path", help="Path in which saving the PDF.")
parser.add_argument("--pages_folder", help = "Where to store screenshots.")


args = parser.parse_args()
if os.path.exists(args.pages_folder):
    shutil.rmtree(args.pages_folder)

if not os.path.exists(args.pages_folder):
    os.makedirs(args.pages_folder)

# Open the PDF file
pdf_document = fitz.open(args.pdf_path)

# Iterate through each page and convert to an image
for page_number in range(pdf_document.page_count):
    # Get the page
    page = pdf_document[page_number]

    # Convert the page to an image
    pix = page.get_pixmap()

    # Create a Pillow Image object from the pixmap
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Save the image
    image.save(f"./{args.pages_folder}/page_{page_number + 1}.png")

# Close the PDF file
pdf_document.close()

class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize(
            (int(round(scale * width)), int(round(scale * height)))
        )

        return resized_image


detection_transform = transforms.Compose(
    [
        MaxResize(800),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

structure_transform = transforms.Compose(
    [
        MaxResize(1000),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    device = torch.device("cpu")
    print("Using CPU")
# load table detection model
# processor = TableTransformerImageProcessor(max_size=800)
model = AutoModelForObjectDetection.from_pretrained(
    "microsoft/table-transformer-detection", revision="no_timm"
).to(device)

# load table structure recognition model
# structure_processor = TableTransformerImageProcessor(max_size=1000)
structure_model = AutoModelForObjectDetection.from_pretrained(
    "microsoft/table-transformer-structure-recognition-v1.1-all"
).to(device)

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    width, height = size
    boxes = box_cxcywh_to_xyxy(out_bbox)
    boxes = boxes * torch.tensor(
        [width, height, width, height], dtype=torch.float32
    )
    return boxes


def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]
    pred_bboxes = [
        elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)
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


def detect_tables_in_cropped_image(cropped_image):
    """
    Detect tables in the cropped image using the model.
    """
    pixel_values = detection_transform(cropped_image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(pixel_values)

    detected_tables = outputs_to_objects(outputs, cropped_image.size, model.config.id2label)
    return len(detected_tables) > 0  # Returns True if any table is detected

def detect_and_crop_save_table(file_path, cropped_table_directory):
    image = Image.open(file_path)
    filename, _ = os.path.splitext(file_path.split("/")[-1])

    if not os.path.exists(cropped_table_directory):
        os.makedirs(cropped_table_directory)

    pixel_values = detection_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(pixel_values)

    id2label = model.config.id2label
    id2label[len(model.config.id2label)] = "no object"
    detected_tables = outputs_to_objects(outputs, image.size, id2label)

    print(f"Number of tables detected in original image: {len(detected_tables)}")

    for idx, table in enumerate(detected_tables):
        cropped_table = image.crop(table["bbox"])
        cropped_table_path = f"./{cropped_table_directory}/{filename}_{idx}.png"
        cropped_table.save(cropped_table_path)

        # Reapply detection on the cropped table
        if detect_tables_in_cropped_image(cropped_table):
            print(f"Table detected in cropped image: {cropped_table_path}")
        else:
            print(f"No table detected in cropped image: {cropped_table_path}")
            os.remove(cropped_table_path)

cropped_table_directory = args.pdf_path.split("/")[1].split(".pdf")[0] + '_tables'
if not os.path.exists(cropped_table_directory):
    os.makedirs(cropped_table_directory)

for file_path in os.listdir(args.pages_folder):
    detect_and_crop_save_table(f'{args.pages_folder}/{file_path}', cropped_table_directory)