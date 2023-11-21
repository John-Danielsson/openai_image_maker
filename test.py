from ImageGenerator import ImageGenerator
from test_prompts import test_prompt1, test_prompt5

if __name__ =="__main__":
    model = "gpt-3.5-turbo"
    # model = "gpt-4"
    imageMaker = ImageGenerator(model)
    test = imageMaker.make_image(
        image_prompt=test_prompt5,
        n_images=2,
        image_size=1
    )
    print(f"test=\"{test}\"")
