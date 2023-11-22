import openai
import os
from dotenv import load_dotenv
from prompt_engineering import main_context, length_restriction, prompt_setup


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class ImageGenerator:
    """
    A class used to generate images based on prompts using OpenAI"s API.

    Attributes
    ----------
    chat_model : str
        The model identifier used for chat completions.
    _client : openai.OpenAI
        Client object to interact with OpenAI"s API.
    _context : list
        The context used for generating prompts.

    Methods
    -------
    make_image(image_prompt, n_images, image_size=0):
        Generates an image based on the provided prompt.

    _upgrade_prompt(prompt_text):
        Creates and refines a prompt for image generation.

    _get_image_data(image_prompt, n_images, image_size):
        Retrieves image data from OpenAI based on the given prompt.

    make_image_with_voice(audio_file_path):
        Generates an image based on a voice transcript.
    """

    def __init__(self, chat_model="gpt-3.5-turbo", image_model="dall-e-2") -> None:
        """
        Constructs all the necessary attributes for the ImageGenerator object.

        Parameters
        ----------
        chat_model : str
            The model identifier to be used for chat completions.
        """
        self.chat_model = chat_model
        self.image_model = image_model
        self._client = openai.OpenAI()
        self.__completions_messages = [
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
        List[str]
            URL of the generated image.
        """
        if n_images < 1:
            raise ValueError("n_images must be at least 1")
        elif self.image_model == "dall-e-3" and n_images > 1:
            raise ValueError("n_images must be 1 when self.image_model == \"dall-e-3\"")
        elif self.image_model == "dall-e-2" and n_images > 10:
            raise ValueError("n_images must be between 1 and 10 (inclusive) when self.image_model == \"dall-e-2\"")
        upgraded_prompt = self._upgrade_prompt(image_prompt)
        image_data = self._get_image_data(
            image_prompt=upgraded_prompt,
            n_images=n_images,
            image_size=image_size
        )
        image_urls = []
        for image in image_data.data:
            image_urls.append(image.url)
        return image_urls

    def _upgrade_prompt(self, prompt_text):
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
        self.__completions_messages[-1]["content"] = f"{prompt_setup} \"{prompt_text}\" {length_restriction}"
        response = self._client.chat.completions.create(
            model=self.chat_model,
            messages=self.__completions_messages
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
        data = self._client.images.generate(
            prompt=image_prompt,
        	n=n_images,
        	size=image_size
        )
        return data

    def make_image_voice(self, audio_file_path, n_images, image_size=0):
        """
        Generates an image based on a voice transcript from an audio file.

        This method transcribes the voice from the specified audio file into text 
        and then generates an image based on this transcription. 

        Parameters
        ----------
        audio_file_path : str
            The file path of the audio file to be transcribed and used for image generation.
        n_images : int
            The number of images to generate based on the voice transcript.
        image_size : int, optional
            The size of the generated image (default is 0).

        Returns
        -------
        result
            The generated image(s) based on the voice transcript.
        """
        audio_transcript = self._transcribe(audio_file_path)
        result = self.make_image_text(
            image_prompt=audio_transcript,
            n_images=n_images,
            image_size=image_size
        )
        return result

    def _transcribe(self, audio_file_path):
        """
        Transcribes the content of an audio file into text.

        This method uses a specific model to transcribe the content of the given audio file. 
        The transcription is used for further processing, like generating prompts for image creation.

        Parameters
        ----------
        audio_file_path : str
            The path to the audio file that needs to be transcribed.

        Returns
        -------
        str
            The transcribed text from the audio file.
        """
        if not self.__valid_audio_file(audio_file_path):
            raise ValueError(f"The file at \"{audio_file_path}\" is not a valid audio file.")
        with open(audio_file_path, "rb") as audio_file:
            transcript = self._client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
            return transcript.text
    
    def __valid_audio_file(self, audio_file_path):
        """
        Validates if the given file path points to a valid audio file.

        This method checks whether the file exists and whether its extension
        is one of the common audio file formats. It"s a basic validation
        based on file existence and extension check.

        Note: This method does not validate the content of the file, and 
        a file passing this validation might still not be a proper audio file.

        Parameters
        ----------
        audio_file_path : str
            The file path to be validated as an audio file.

        Returns
        -------
        bool
            True if the file is a valid audio file, False otherwise.
        """
        if not os.path.exists(audio_file_path):
            return False
        _, ext = os.path.splitext(audio_file_path)
        return ext.lower() in ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]

    def set_chat_model(self, chat_model):
        """
        Sets the chat model identifier to be used for chat completions.

        Parameters
        ----------
        chat_model : str
            The model identifier (e.g., "gpt-3.5-turbo") to be used for chat completions.
        """
        self.chat_model = chat_model

    def set_image_model(self, image_model):
        """
        Sets the image model identifier to be used for generating images.

        Parameters
        ----------
        image_model : str
            The model identifier (e.g., "dall-e-2", "dall-e-3") to be used for image generation.
        """
        self.image_model = image_model
