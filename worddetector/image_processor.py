from worddetector.models import Field
import fire

FIELD_SIZE = 5


def get_letters(image_name):
    field = Field(n=FIELD_SIZE, image_name=image_name)
    return field.flatten()


if __name__ == "__main__":
    fire.Fire()
