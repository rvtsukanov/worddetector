from worddetector.models import Field
from pathlib import Path
import os


def test_field():
    field = Field(n=5, image_name=str(Path(os.getcwd()) / Path('assets/png_screen.png')))
    assert len(field.active_cells) == 25
    assert "ЕТИСТЛНЕЙЫФРУННИНММЕАФИГД" == field.one_row()
