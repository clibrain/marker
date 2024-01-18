import os
from PIL import Image, ImageDraw, ImageFont
import shutil

class ImageComposer:
    def __init__(self, images_dir, pdf_path, image_width=600, inner_images_per_image=2):
        self.images_dir = images_dir
        self.pdf_path = pdf_path
        self.image_width = image_width
        self.inner_images_per_image = inner_images_per_image
        self.init_sizes()

    def init_sizes(self):
        # Initialize sizes and ratios based on the provided image_width
        
        inner_margin_width_ratio = 600 / 4
        self.inner_margin_width = int(self.image_width / inner_margin_width_ratio)
        self.inner_images_width = self.image_width - (self.inner_margin_width * 2)

        margin_big_heigth_ratio = 600 / 20
        self.margin_big_heigth = int(self.image_width / margin_big_heigth_ratio)
        self.margin_small_heigth = int(self.margin_big_heigth / 2)

        title_image_margin_ratio = 600 / 6
        self.title_image_margin = int(self.image_width / title_image_margin_ratio)

        title_font_size_ratio = 600 / 18
        self.title_font_size = int(self.image_width / title_font_size_ratio)

    def compose_images(self):
        index = 0
        images = [{}]
        for file_name in os.listdir(self.images_dir):
            try:
                current_image = Image.open(f"{self.images_dir}/{file_name}")
            except Exception:
                continue

            # Resize current image
            current_image_size = self.resize_image(current_image)

            # Create title image
            title_image = self.create_title_image(file_name, current_image_size)

            images[index][file_name] = {
                "image": current_image,
                "title_image": title_image
            }

            if len(images[index]) >= self.inner_images_per_image:
                index += 1
                images.append({})

        self.save_images(images)

    def resize_image(self, image):
        current_image_size = image.size
        current_image_ratio = self.inner_images_width / current_image_size[0]
        current_image_width = int(current_image_size[0] * current_image_ratio)
        current_image_height = int(current_image_size[1] * current_image_ratio)
        resized_size = (current_image_width, current_image_height)

        return image.resize(resized_size, Image.Resampling.LANCZOS)

    def create_title_image(self, file_name, image_size):
        file_name_no_ext = os.path.splitext(file_name)[0]
        font = ImageFont.load_default(self.title_font_size)

        title_image_tmp = Image.new("RGB", (self.image_width, self.image_width), color="yellow")
        title_image_draw_tmp = ImageDraw.Draw(title_image_tmp)
        _, _, width, height = title_image_draw_tmp.textbbox((0, 0), file_name_no_ext, font=font)
        title_image_width = width + (self.title_image_margin * 2)
        title_image_height = height + (self.title_image_margin * 2)

        title_image = Image.new("RGB", (title_image_width, title_image_height), color="yellow")
        title_image_draw = ImageDraw.Draw(title_image)
        title_image_position = ((title_image_width - width) / 2, (title_image_height - height) / 2)
        title_image_draw.text(title_image_position, file_name_no_ext, font=font, fill="red")

        return title_image

    def save_images(self, images):
    
        shutil.rmtree(f"{os.path.splitext(self.pdf_path)[0]}_tables", ignore_errors=True)
        os.makedirs(f"{os.path.splitext(self.pdf_path)[0]}_tables", exist_ok=True)
        
        file_name_template = f"{os.path.splitext(self.pdf_path)[0]}_tables/{{part}}.png"
        for index, images_dict in enumerate(images):
            image_height = self.calculate_image_height(images_dict)
            image = Image.new("RGB", (self.image_width, image_height), color="gray")

            self.paste_images(image, images_dict)

            image.save(file_name_template.format(part=index))

    def calculate_image_height(self, images_dict):
        image_height = self.margin_big_heigth
        for file_name in images_dict:
            title_image_height = images_dict[file_name]["title_image"].size[1]
            inner_image_height = images_dict[file_name]["image"].size[1]
            image_height += self.margin_big_heigth + title_image_height + self.margin_small_heigth + inner_image_height
        image_height += self.margin_big_heigth
        return image_height

    def paste_images(self, base_image, images_dict):
        current_y = self.margin_big_heigth
        for file_name in images_dict:
            base_image.paste(images_dict[file_name]["title_image"], (self.inner_margin_width, current_y))
            current_y += images_dict[file_name]["title_image"].size[1] + self.margin_small_heigth
            base_image.paste(images_dict[file_name]["image"], (self.inner_margin_width, current_y))
            current_y += images_dict[file_name]["image"].size[1] + self.margin_big_heigth


