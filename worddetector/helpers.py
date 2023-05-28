import numpy as np
import cv2


def match_k_blobs(array, k, threshold, key):
    if array is None or len(array) < k or k < 2:
        return

    array = list(zip(array, range(len(array))))
    array = sorted(array, key=lambda x: key(x[0]))  # O(nlogn)

    output = [array[0]]

    i = 1
    while i < len(array):
        if (
            np.abs(key(array[i][0]) - key(output[-1][0]))
            < key(output[-1][0]) * threshold
        ):
            output.append(array[i])
            if len(output) == k:
                return np.array([item[1] for item in output], dtype=np.int16).astype(
                    int
                )

            i += 1
        else:
            output = [array[i]]
            if len(output) == k:
                return np.array([item[1] for item in output], dtype=np.int16).astype(
                    int
                )
            i += 1


def show(img):
    cv2.imshow("Contours", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
