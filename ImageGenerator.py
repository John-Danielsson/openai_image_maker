import openai

client = openai.OpenAI()
import os
from dotenv import load_dotenv
from prompt_engineering import main_context, length_restriction, prompt_setup

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class ImageGenerator:
    """
    A class used to generate images based on prompts using OpenAI's API.

    Attributes
    ----------
    chat_model : str
        The model identifier used for chat completions.
    _client : openai.OpenAI
        Client object to interact with OpenAI's API.
    _context : list
        The context used for generating prompts.

    Methods
    -------
    make_image(image_prompt, n_images, image_size=0):
        Generates an image based on the provided prompt.

    _make_prompt(prompt_text):
        Creates and refines a prompt for image generation.

    _get_image_data(image_prompt, n_images, image_size):
        Retrieves image data from OpenAI based on the given prompt.

    make_image_with_voice(audio_file_path):
        Generates an image based on a voice transcript.
    """

    def __init__(self, chat_model) -> None:
        """
        Constructs all the necessary attributes for the ImageGenerator object.

        Parameters
        ----------
        chat_model : str
            The model identifier to be used for chat completions.
        """
        self.chat_model = chat_model
        self._client = openai.OpenAI()
        self._context = [
            {"role": "system", "content": main_context},
            {"role": "user", "content": ""}
        ]

    def make_image_text(self, image_prompt, n_images, image_size=0):
        """
        Generates an image based on the provided prompt.

        Parameters
        ----------
        image_prompt : str
            The prompt to be used for generating the image.
        n_images : int
            The number of images to generate.
        image_size : int, optional
            The size of the generated image (default=0).
            Mapping and Range:
                0: "256x256"
                1: "512x512"
                2: "1024x1024"

        Returns
        -------
        str
            URL of the generated image.
        """
        upgraded_prompt = self._make_prompt(image_prompt)
        image_data = self._get_image_data(upgraded_prompt, n_images, image_size)
        image_urls = []
        for image in image_data.data:
            image_urls.append(image.url)
        return image_urls

    def _make_prompt(self, prompt_text):
        """
        Creates and refines a prompt for image generation.

        Parameters
        ----------
        prompt_text : str
            The text of the prompt.

        Returns
        -------
        str
            The refined prompt.
        """
        self._context[-1]["content"] = f"{prompt_setup} \"{prompt_text}\" {length_restriction}"
        print("\n\nself._context[-1][\"content\"]={}\n\n".format(self._context[-1]["content"]))
        response = self._client.chat.completions.create(
            model=self.chat_model,
            # response_format={"type": "json_object"},
            messages=self._context
        )
        return response.choices[0].message.content

    def _get_image_data(self, image_prompt, n_images, image_size):
        """
        Retrieves image data from OpenAI based on the given prompt.

        Parameters
        ----------
        image_prompt : str
            The prompt used for image generation.
        n_images : int
            The number of images to generate.
        image_size : int
            The size of the generated image.

        Returns
        -------
        dict
            The data of the generated images.
        """
        data = client.images.generate(
            prompt=image_prompt,
        	n=n_images,
        	size=f"{256 * (1 << image_size)}x{256 * (1 << image_size)}"
        )
        return data

    def make_image_voice(self, audio_file_path, n_images, image_size=0):
        """
        Generates an image based on a voice transcript.

        Parameters
        ----------
        audio_file_path : str
            The file path of the audio file to be transcribed.

        Returns
        -------
        str
            The transcript of the audio file.
        """
        prompt_text = self._transcribe(audio_file_path)
        print(f"\n\nprompt_text={prompt_text}\n\n")
        result = self.make_image_text(
            prompt_text,
            n_images,
            image_size
        )
        return result

    def _transcribe(self, audio_file_path):
        audio_file = open(audio_file_path, "rb")
        transcript = self._client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
        print("\n\n_transcribe()")
        print("transcript={transcript}\n\n")
        return transcript.text
