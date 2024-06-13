from PIL import Image
from io import BytesIO
import base64
import fitz

def convert_pdf_to_images(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        images = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        return images
    except Exception as e:
        return f"Error during PDF conversion: {e}"

def combine_images(images, orientation='vertical'):
    widths, heights = zip(*(i.size for i in images))
    if orientation == 'vertical':
        total_width = max(widths)
        total_height = sum(heights)
        combined_image = Image.new('RGB', (total_width, total_height))
        y_offset = 0
        for img in images:
            combined_image.paste(img, (0, y_offset))
            y_offset += img.height
    else:
        total_width = sum(widths)
        total_height = max(heights)
        combined_image = Image.new('RGB', (total_width, total_height))
        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.width
    return combined_image

def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
