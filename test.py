from ImageGenerator import ImageGenerator
from test_prompts import test_prompt1, test_prompt5
import openai
import asyncio

async def main():
    chat_model = "gpt-3.5-turbo"
    # chat_model = "gpt-4"
    image_model = "dall-e-2"
    # image_model = "dall-e-3"
    client = openai.AsyncOpenAI()

    imageMaker = ImageGenerator(
        chat_model=chat_model,
        image_model=image_model,
        client=client
    )

    # Uncomment to test text-based image generation
    # test_text = await imageMaker.make_image_text(
    #     image_prompt=test_prompt5,
    #     n_images=2,
    #     image_size="1024x1024"
    # )
    # print(f"test_text=\"{test_text}\"")

    # Test audio-based image generation
    audio_file_path = "voice_prompts/test_voice_prompt_1.m4a"
    image_urls = await imageMaker.make_image_voice(
        audio_file_path=audio_file_path,
        n_images=2,
        image_size="256x256"
    )
    for i, url in enumerate(image_urls):
        print(f"\nurl {i}={url}\n")

if __name__ == "__main__":
    asyncio.run(main())
