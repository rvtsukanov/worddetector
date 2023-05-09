import cv2
import pytesseract
import enum
import numpy as np
from helpers import match_k_blobs, show
import re

class PolygonAngle(enum.Enum):
    SQUARE = 4
    SEMISQUARE = 5

class ScreenShotThreshold(enum.Enum):
    IPHONE_OR_IPAD = 0.01
    PHOTO = 0.1

BONUS_SQUARES = 4
TOTAL_SQUARES = 25


class Cell:
    'One cell of the Field'

    bw_screenshot = None
    prefix = 'Ord'
    color = (0, 255, 0)

    def __init__(self, contour):
        self.contour = contour
        self.set_polygon()
        self.center = self.build_center()
        self.letter = None

    def __lt__(self, other):
        if isinstance(other, Cell):
            return self.perimeter < other.perimeter
        if isinstance(other, int):
            return self.perimeter < other

    def __gt__(self, other):
        if isinstance(other, Cell):
            return self.perimeter > other.perimeter
        if isinstance(other, int):
            return self.perimeter > other

    def __sub__(self, other):
        if isinstance(other, Cell):
            return self.perimeter - other.perimeter
        if isinstance(other, int):
            return self.perimeter - other

    def __add__(self, other):
        if isinstance(other, Cell):
            return self.perimeter + other.perimeter
        if isinstance(other, int):
            return self.perimeter + other

    def __mul__(self, other):
        if isinstance(other, Cell):
            return self.perimeter * other.perimeter
        if isinstance(other, int) or isinstance(other,  float):
            return self.perimeter * other

    def __repr__(self):
        return f'{self.prefix}[{self.perimeter:.2f}][{self.letter}]'

    def render(self, img):
        cv2.drawContours(img, [self.contour], -1, self.color, 10)
        cv2.circle(img, self.center, radius=5, color=(0, 0, 255), thickness=4)

    def build_center(self):
        moments = cv2.moments(self.polygon)
        return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])

    @property
    def dim(self):
        return self.polygon.shape[0]

    @property
    def perimeter(self):
        return cv2.arcLength(self.contour, True)

    @classmethod
    def build_polygon(cls, contour):
        return cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

    def set_polygon(self):
        self.polygon = self.build_polygon(self.contour)

    def crop_img_to_roi(self, img):
        x, y, w, h = cv2.boundingRect(self.contour)
        return img[y+6:y+h-6, x+6:x+h-6]  # TODO: make proportional vignette

    def detect_letter(self):
        cropped_image = self.crop_img_to_roi(self.bw_screenshot)
        answer = pytesseract.image_to_string(cropped_image, lang="rus", config='--psm 10 --oem 3')
        self.letter = re.sub('\W+','', answer.upper())
        return answer


class BonusCell(Cell):
    prefix = 'Bonus'
    color = (255, 255, 0)

    def __init__(self, contour):
        super().__init__(contour)


class Field: # TODO: preprocess with gray theme
    def __init__(self, n, image_name):
        self.n = n
        self.image = cv2.imread(image_name)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # show(self.gray)
        # Because background is true white -- improve!
        _, self.thresholded_image = cv2.threshold(self.gray, 254, 255, 0)
        contours, _ = cv2.findContours(self.thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        set_of_cells = self.preprocess_contours(contours)
        self.active_cells = set_of_cells
        self.detect_letters()

        for cell in self.active_cells:
            cell.render(self.image)

        show(self.image)


    def detect_letters(self):
        for cell in self.active_cells:
            cell.detect_letter()

    def preprocess_contours(self, contours):
        perimeters = []
        for cnt in contours:
            poly = Cell.build_polygon(cnt)

            if len(poly) == PolygonAngle.SQUARE.value:
                cell = Cell(cnt)
                cell.bw_screenshot = self.gray
                # cell.detect_letter()
                perimeters.append(cell)

            if len(poly) == PolygonAngle.SEMISQUARE.value:
                cell = BonusCell(cnt)
                cell.bw_screenshot = self.gray
                # cell.detect_letter()
                perimeters.append(cell)

        bonus_idx = match_k_blobs(perimeters, BONUS_SQUARES, threshold=ScreenShotThreshold.IPHONE_OR_IPAD.value)
        ordinary_idx = match_k_blobs(perimeters, TOTAL_SQUARES - BONUS_SQUARES,
                                     threshold=ScreenShotThreshold.IPHONE_OR_IPAD.value)

        return np.array(perimeters)[np.concatenate([bonus_idx, ordinary_idx])]


    def create_active_cells_from_polygons(self, list_of_polygons):
        pass



