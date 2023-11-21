import openai
import os
from dotenv import load_dotenv
from prompt_engineering import PROMPT, PROMPT_PRIMER


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class ImageGenerator:

	def __init__(self, chat_model) -> None:
		self.chat_model = chat_model
		self._client = openai.OpenAI()
		self._context = []
		for p in PROMPT_PRIMER:
			self._context.append({"role": "system", "content": p})
		self._context.append({"role": "user", "content": ""})

	def make_image(self, image_prompt, n_images, image_size=0):
		"""
		The make_image() function allows you to create an original image with Open AI"s API
		given a text prompt.
		Generated images can have a size of 256x256 (image_size=0), 512x512 (image_size=1),
		or 1024x1024 (image_size=2) pixels.
		Smaller sizes are faster to generate. You can request 1-10 images at a time using
		the n parameter.
		"""
		upgraded_prompt = self._make_prompt(image_prompt)
		image_data = self._get_image_data(upgraded_prompt, n_images, image_size)
		return image_data["data"][0]["url"]

	def _make_prompt(self, prompt_text):
		self._context[-1]["content"] = prompt_text
		response = self._client.chat.completions.create(
			model=self.chat_model,
			messages=self._context
		)
		return response["choices"][0]["message"]["content"]

	def _get_image_data(self, image_prompt, n_images, image_size):
		data = openai.Image.create(
			prompt=image_prompt,
			n=n_images,
			size=f"{256 * (1 << image_size)}x{256 * (1 << image_size)}"
		)
		return data

	def make_image_with_voice(self, audio_file_path):
		audio_file = open(audio_file_path, "rb")
		transcript = self._client.audio.translations.create(
		model="whisper-1", 
		file=audio_file
		)
		return transcript["text"]