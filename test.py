from ImageGenerator import ImageGenerator
from test_prompts import test_prompt1, test_prompt5

if __name__ =="__main__":
    model = "gpt-3.5-turbo"
    # model = "gpt-4"
    imageMaker = ImageGenerator(model)

    # test_text = imageMaker.make_image_text(
    #     image_prompt=test_prompt5,
    #     n_images=2,
    #     image_size=1
    # )
    # print(f"test_text=\"{test_text}\"")

    audio_file_path = "voice_prompts/example.m4a"
    test_voice = imageMaker.make_image_voice(
        audio_file_path=audio_file_path,
        n_images=2,
        image_size=1
    )
    print(f"test_voice=\"{test_voice}\"")
