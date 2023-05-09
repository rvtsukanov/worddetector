import cv2
from models import Cell, Field
FIELD_SIZE = 5


field = Field(n=FIELD_SIZE, image_name='iphone.png')
print(field.active_cells)


# img = cv2.imread('iphone.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ordinary_one_cell = np.array(perimeters)[ordinary_idx.tolist()][0]
# ordinary_one_cell.render(img)

# cropped = ordinary_one_cell.crop_img_to_roi(thresh)
# letter = ordinary_one_cell.detect_letter(thresh)
# print(letter)

# bonus_one_cell = np.array(perimeters)[bonus_idx.tolist()][0]
# bonus_one_cell.render(img)

# cv2.imshow('Contours', cropped)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# rest_idx = match_k_blobs([p[0] for p in perimeters], TOTAL_SQUARES - BONUS_SQUARES, 0.01)
#
# print(perimeters)
# perimeters = np.array(perimeters)
#
# print(bonus_idx)
#
# bonus_contours = [p[1] for p in perimeters[bonus_idx.astype(int)]]
# rest_contours = [p[1] for p in perimeters[rest_idx.astype(int)]]


# cv2.drawContours(img, bonus_contours, -1, (0, 255, 0), 3)
# cv2.drawContours(img, rest_contours, -1, (0, 255, 0), 3)
# #
# cv2.imshow('Contours', img)
# # cv2.imshow('Contours', thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(gray)
# print(gray.shape)
# print(gray.max(), gray.min())

# cv2.imwrite('greyscale.png', gray)