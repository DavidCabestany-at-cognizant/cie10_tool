import pytest
from PIL import Image
from ..src.utils.image_utils import convert_pdf_to_images, combine_images, encode_image

def test_convert_pdf_to_images():
    pdf_path = "./sample_diagnose/sample.pdf" 
    images = convert_pdf_to_images(pdf_path)
    assert isinstance(images, list)
    assert all(isinstance(img, Image.Image) for img in images)

def test_combine_images():
    img1 = Image.new('RGB', (100, 200), color = 'red')
    img2 = Image.new('RGB', (100, 200), color = 'blue')
    combined_image = combine_images([img1, img2], orientation='vertical')
    assert combined_image.size == (100, 400)

    combined_image = combine_images([img1, img2], orientation='horizontal')
    assert combined_image.size == (200, 200)

def test_encode_image():
    img = Image.new('RGB', (100, 200), color = 'red')
    encoded_image = encode_image(img)
    assert isinstance(encoded_image, str)
    assert encoded_image.startswith('/9j/')  
