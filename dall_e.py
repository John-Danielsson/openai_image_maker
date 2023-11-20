import openai
import os
from dotenv import load_dotenv


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def make_image(image_prompt, n_images, image_size=0):
	"""
	The make_image() function allows you to create an original image with Open AI's API
	given a text prompt.
	Generated images can have a size of 256x256 (image_size=0), 512x512 (image_size=1),
	or 1024x1024 (image_size=2) pixels.
	Smaller sizes are faster to generate. You can request 1-10 images at a time using
	the n parameter.
	"""
	response = openai.Image.create(
		prompt=image_prompt,
		n=n_images,
		size=f"{256 * (1 << image_size)}x{256 * (1 << image_size)}"
	)
	return response['data'][0]['url']
