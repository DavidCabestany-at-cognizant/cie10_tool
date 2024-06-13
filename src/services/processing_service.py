from ..utils.image_utils import convert_pdf_to_images, combine_images
from ..utils.gpt_utils import send_image_to_gpt

def process_and_combine_images(pdf_file_path, orientation, config):
    images = convert_pdf_to_images(pdf_file_path)
    if isinstance(images, str) and images.startswith("Error"):
        return images, []
    combined_image = combine_images(images, orientation)
    marked_fields = send_image_to_gpt(combined_image, config)
    if "Error" in marked_fields:
        return marked_fields, []
    return marked_fields, [combined_image]
