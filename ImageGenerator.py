import openai
import os
from dotenv import load_dotenv
from prompt_engineering import PROMPT, PROMPT_PRIMER


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class ImageGenerator:

	def __init__(self, model) -> None:
		self.model = model
		self.__client = openai.OpenAI()
		self.__context = []
		for p in PROMPT_PRIMER:
			self.__context.append({"role": "system", "content": p})
		self.__context.append({"role": "user", "content": ""})

	def make_image(self, image_prompt, n_images, image_size=0):
		"""
		The make_image() function allows you to create an original image with Open AI"s API
		given a text prompt.
		Generated images can have a size of 256x256 (image_size=0), 512x512 (image_size=1),
		or 1024x1024 (image_size=2) pixels.
		Smaller sizes are faster to generate. You can request 1-10 images at a time using
		the n parameter.
		"""
		better_image_prompt = self.make_prompt(image_prompt)
		image_data = self.get_data(better_image_prompt, n_images, image_size)
		return image_data["data"][0]["url"]

	def make_prompt(self, prompt_text):
		self.__context[-1]["content"] = prompt_text
		response = self.__client.chat.completions.create(
			model=self.model,
			messages=self.__context
		)
		return response["choices"][0]["message"]["content"]

	def get_data(self, image_prompt, n_images, image_size):
		data = openai.Image.create(
			prompt=image_prompt,
			n=n_images,
			size=f"{256 * (1 << image_size)}x{256 * (1 << image_size)}"
		)
		return data