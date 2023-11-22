from ImageGenerator import ImageGenerator
from test_prompts import test_prompt1, test_prompt5
import openai

if __name__ =="__main__":
    chat_model = "gpt-3.5-turbo"
    # chat_model = "gpt-4"
    image_model = "dall-e-2"
    # image_model = "dall-e-3"
    client  = openai.OpenAI()

    imageMaker = ImageGenerator(
        chat_model=chat_model,
        image_model=image_model,
        client=client
    )

    # test_text = imageMaker.make_image_text(
    #     image_prompt=test_prompt5,
    #     n_images=2,
    #     image_size=1
    # )
    # print(f"test_text=\"{test_text}\"")
    audio_file_path = "voice_prompts/test_voice_prompt_1.m4a"
    image_urls = imageMaker.make_image_voice(
        audio_file_path=audio_file_path,
        n_images=2,
        image_size="256x256"
    )
    for i in range(len(image_urls)):
        print(f"url {i}={image_urls[i]}\n")
