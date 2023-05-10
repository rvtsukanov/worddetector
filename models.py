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
FIELD_SIZE = 5

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

    def render(self):
        cv2.drawContours(self.bw_screenshot, [self.contour], -1, self.color, 10)
        cv2.circle(self.bw_screenshot, self.center, radius=5, color=(0, 0, 255), thickness=4)
        # cropped_image = self.crop_img_to_roi(self.bw_screenshot)
        # show(cropped_image)

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

    def build_roi(self):
        x, y, w, h = cv2.boundingRect(self.contour)
        PIXEL_OFFSET = 10
        self.roi = self.bw_screenshot[y+PIXEL_OFFSET:y+h-PIXEL_OFFSET,
                   x+PIXEL_OFFSET:x+h-PIXEL_OFFSET]  # TODO: make proportional vignette

    def detect_letter(self):
        self.build_roi()
        answer = pytesseract.image_to_string(self.roi, lang="rus", config='--psm 10 --oem 3')
        self.letter = re.sub('\W+','', answer.upper())
        return answer


class BonusCell(Cell):
    prefix = 'Bonus'
    color = (255, 255, 0)

    def __init__(self, contour):
        super().__init__(contour)

    def build_roi(self):
        # https://stackoverflow.com/questions/48301186/cropping-concave-polygon-from-image-using-opencv-python

        rect = cv2.boundingRect(self.contour)
        x, y, w, h = rect
        croped = self.bw_screenshot[y:y + h, x:x + w].copy()
        pts = self.contour - self.contour.min(axis=0)

        mask = np.zeros(croped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

        # mask = cv2.bitwise_not(mask)

        dst = cv2.bitwise_and(croped, croped, mask=mask)
        # dst_inv = cv2.bitwise_not(croped, croped, mask=mask)
        bg = np.ones_like(croped, np.uint8) * 255
        cv2.bitwise_not(bg, bg, mask=mask)
        dst2 = bg + dst
        self.roi = dst2




class Field: # TODO: preprocess with gray theme
    def __init__(self, n, image_name):
        self.n = n
        self.image = cv2.imread(image_name)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.active_cells_map = []

        # show(self.gray)
        # Because background is true white -- improve!
        _, self.thresholded_image = cv2.threshold(self.gray, 254, 255, 0)
        contours, _ = cv2.findContours(self.thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        set_of_cells = self.preprocess_contours(contours)

        print(set_of_cells)

        img = cv2.drawContours(self.image, [c.polygon for c in set_of_cells], -1, (255, 0, 0), 10)
        show(img)
        self.active_cells = set_of_cells
        self.detect_letters()
        self.reindex_cells()

    def flatten(self):
        output = []
        for row in self.active_cells_map:
            flat_row = []
            for lt in row:
                flat_row.append(lt.letter)

            output.append(flat_row)
        return np.array(output)


    def reindex_cells(self):
        sorted_points = np.array(sorted(self.active_cells, key=lambda x: x.center[1]))
        rows = []
        for i in range(FIELD_SIZE):
            one_row = sorted_points[i * FIELD_SIZE:(i + 1) * FIELD_SIZE]
            one_row = np.array(sorted(one_row, key=lambda x: x.center[0]))
            rows.append(one_row)
        self.active_cells_map = rows

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
                perimeters.append(cell)

            if len(poly) == PolygonAngle.SEMISQUARE.value:
                cell = BonusCell(cnt)
                cell.bw_screenshot = self.gray
                perimeters.append(cell)

        bonus_idx = match_k_blobs(perimeters, BONUS_SQUARES, threshold=ScreenShotThreshold.IPHONE_OR_IPAD.value)
        ordinary_idx = match_k_blobs(perimeters, TOTAL_SQUARES - BONUS_SQUARES,
                                     threshold=ScreenShotThreshold.IPHONE_OR_IPAD.value)
        print(bonus_idx)
        print(sorted(np.array(perimeters)))
        # print(ordinary_idx)
        return np.array(perimeters)[np.concatenate([bonus_idx, ordinary_idx])]
        # return np.array(perimeters)



