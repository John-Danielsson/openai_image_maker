import openai
import os
from dotenv import load_dotenv
from prompt_engineering import PROMPT, PROMPT_PRIMER

client = openai.OpenAI()

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def make_image(model, image_prompt, n_images, image_size=0):
	"""
	The make_image() function allows you to create an original image with Open AI"s API
	given a text prompt.
	Generated images can have a size of 256x256 (image_size=0), 512x512 (image_size=1),
	or 1024x1024 (image_size=2) pixels.
	Smaller sizes are faster to generate. You can request 1-10 images at a time using
	the n parameter.
	"""
	updated_image_prompt = make_prompt(model, image_prompt)
	image_data = get_data(updated_image_prompt, n_images, image_size)
	return image_data["data"][0]["url"]

def make_prompt(model, prompt_text):
	context = []
	for i in range(len(PROMPT_PRIMER)):
		context.append({"role": "system", "content": PROMPT_PRIMER[i]})
	context.append({"role": "user", "content": PROMPT.format(prompt_text)})
	response = client.chat.completions.create(
		model=model,
		messages=context
	)
	return response["choices"][0]["message"]["content"]

def get_data(image_prompt, n_images, image_size):
	data = openai.Image.create(
		prompt=image_prompt,
		n=n_images,
		size=f"{256 * (1 << image_size)}x{256 * (1 << image_size)}"
	)
	return data