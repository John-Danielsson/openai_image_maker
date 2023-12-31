import openai
import os
from prompt_engineering import main_context, length_restriction, prompt_setup


class ImageGenerator:
    """
    A class for generating images using OpenAI's API, with capabilities for both text and audio-based prompts asynchronously.

    This class interfaces with OpenAI's chat and image models to generate images based on textual or spoken prompts asynchronously.
    It supports various configurations for image generation and includes methods for refining prompts and handling audio inputs asynchronously.

    Attributes
    ----------
    chat_model : str
        The identifier of the chat model used for text prompt completions (e.g., 'gpt-3.5-turbo').
    image_model : str
        The identifier of the image generation model used (e.g., 'dall-e-2').
    _client : openai.AsyncOpenAI
        The OpenAI client instance for asynchronous API interactions.

    Methods
    -------
    make_image_text(image_prompt, n_images, image_size):
        Asynchronously generates images based on a text prompt.
    _upgrade_prompt(prompt_text):
        Asynchronously refines a prompt text for improved image generation.
    _get_image_data(image_prompt, n_images, image_size):
        Asynchronously retrieves image data from OpenAI API using the given prompt.
    make_image_voice(audio_file_path, n_images, image_size):
        Asynchronously generates images based on the transcript of an audio file.
    _transcribe(audio_file_path):
        Asynchronously transcribes an audio file into text.
    __valid_audio_file(audio_file_path):
        Validates if a file path points to a valid audio file.
    set_chat_model(chat_model):
        Updates the chat model identifier.
    set_image_model(image_model):
        Updates the image model identifier.
    """

    def __init__(self, chat_model: str, image_model: str, client: openai.AsyncOpenAI) -> None:
        """
        Initializes the ImageGenerator with specified chat and image models for asynchronous operations.

        Parameters
        ----------
        chat_model : str
            The model identifier for chat completions (default is 'gpt-3.5-turbo').
        image_model : str
            The model identifier for image generation (default is 'dall-e-2').
        client: openai.AsyncOpenAI
            The OpenAI client instance for asynchronous API interactions.
        """
        self.chat_model = chat_model
        self.image_model = image_model
        self._client = client
        self.__completions_messages = [
            {"role": "system", "content": main_context},
            {"role": "user", "content": ""}
        ]

    async def make_image_text(self, image_prompt: str, n_images: int, image_size: str) -> list:
        """
        Asynchronously generates images based on a provided text prompt.

        Uses the given prompt to asynchronously generate a specified number of images of a certain size.
        Includes validation for the number of images based on the selected model.

        Parameters
        ----------
        image_prompt : str
            The text prompt to use for image generation.
        n_images : int
            The number of images to generate.
        image_size : str
            The size of the generated images (e.g., '1024x1024').

        Returns
        -------
        List[str]
            A list of URLs pointing to the generated images.

        Raises
        ------
        ValueError
            If n_images is not within the allowed range for the selected image model.
        """
        if n_images < 1:
            raise ValueError("n_images must be at least 1")
        elif self.image_model == "dall-e-3" and n_images > 1:
            raise ValueError("n_images must be 1 when self.image_model == \"dall-e-3\"")
        elif self.image_model == "dall-e-2" and n_images > 10:
            raise ValueError("n_images must be between 1 and 10 (inclusive) when self.image_model == \"dall-e-2\"")
        upgraded_prompt = await self._upgrade_prompt(image_prompt)
        image_data = await self._get_image_data(
            image_prompt=upgraded_prompt,
            n_images=n_images,
            image_size=image_size
        )
        image_urls = [image.url for image in image_data.data]
        return image_urls

    async def _upgrade_prompt(self, prompt_text: str) -> str:
        """
        Asynchronously enhances a text prompt to improve the quality of generated images.

        Uses the chat model to asynchronously refine a given prompt, making it more suitable for image generation.

        Parameters
        ----------
        prompt_text : str
            The original prompt text to be refined.

        Returns
        -------
        str
            The refined prompt text.
        """
        self.__completions_messages[-1]["content"] = f"{prompt_setup} \"{prompt_text}\" {length_restriction}"
        response = await self._client.chat.completions.create(
            model=self.chat_model,
            messages=self.__completions_messages
        )
        return response.choices[0].message.content

    async def _get_image_data(self, image_prompt: str, n_images: int, image_size: str) -> dict:
        """
        Asynchronously fetches image data from OpenAI API based on a given prompt.

        Retrieves image data using the OpenAI API, given a prompt, the number of images, and their size.

        Parameters
        ----------
        image_prompt : str
            The prompt used for image generation.
        n_images : int
            The number of images to retrieve.
        image_size : str
            The size of the images to be generated (e.g., '1024x1024').

        Returns
        -------
        dict
            A dictionary containing data of the generated images.
        """
        data = await self._client.images.generate(
            prompt=image_prompt,
            n=n_images,
            size=image_size
        )
        return data

    async def make_image_voice(self, audio_file_path: str, n_images: int, image_size: str) -> list:
        """
        Asynchronously generates images based on the transcription of an audio file.

        Transcribes the content of the provided audio file and then uses the transcript as a prompt for image generation.
        Handles both the transcription and image generation processes asynchronously.

        Parameters
        ----------
        audio_file_path : str
            The file path of the audio file to be transcribed.
        n_images : int
            The number of images to generate from the transcript.
        image_size : str
            The size of the generated images (e.g., '1024x1024').

        Returns
        -------
        list[str]
            A list of URLs to the generated images.

        Raises
        ------
        ValueError
            If the provided audio file is not valid.
        """
        audio_transcript = await self._transcribe(audio_file_path)
        image_urls = await self.make_image_text(
            image_prompt=audio_transcript,
            n_images=n_images,
            image_size=image_size
        )
        return image_urls

    async def _transcribe(self, audio_file_path: str) -> str:
        """
        Asynchronously transcribes the content of an audio file into text.

        Uses OpenAI's transcription model to convert audio content into a text transcript asynchronously.
        Primarily used for generating image prompts from spoken content.

        Parameters
        ----------
        audio_file_path : str
            The path to the audio file to be transcribed.

        Returns
        -------
        str
            The transcribed text from the audio file.

        Raises
        ------
        ValueError
            If the audio file is not valid or cannot be transcribed.
        """
        if not self.__valid_audio_file(audio_file_path):
            raise ValueError(f"The file at '{audio_file_path}' is not a valid audio file.")
        with open(audio_file_path, "rb") as audio_file:
            transcript = await self._client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
            return transcript.text

    def __valid_audio_file(self, audio_file_path: str) -> bool:
        """
        Checks if a given file path points to a valid audio file.

        This internal method validates the existence and format of an audio file. It's a basic check based on file extension.

        Parameters
        ----------
        audio_file_path : str
            The file path to be checked.

        Returns
        -------
        bool
            True if the file is a valid audio file, False otherwise.
        """
        if not os.path.exists(audio_file_path):
            return False
        _, ext = os.path.splitext(audio_file_path)
        ext = ext[1:].lower()
        return ext in ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]

    def set_chat_model(self, chat_model: str) -> None:
        """
        Sets a new chat model identifier for text prompt completions.

        This method updates the chat model used by the ImageGenerator instance.

        Parameters
        ----------
        chat_model : str
            The new chat model identifier to be set.
        """
        self.chat_model = chat_model

    def set_image_model(self, image_model: str) -> None:
        """
        Sets a new image model identifier for image generation.

        This method updates the image model used by the ImageGenerator instance.

        Parameters
        ----------
        image_model : str
            The new image model identifier to be set.
        """
        self.image_model = image_model
