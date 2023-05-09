import numpy as np
import cv2

def match_k_blobs(array, k, threshold):
    # print(f'Array: {array}')
    if array is None or len(array) < k or k < 2:
        return

    array = list(zip(array, range(len(array))))
    array = sorted(array, key=lambda x: x[0])  # O(nlogn)
    output = [array[0]]

    i = 1
    while i < len(array):
        if np.abs(array[i][0] - output[-1][0]) < output[-1][0] * threshold:
            output.append(array[i])
            if len(output) == k:
                return np.array([item[1] for item in output], dtype=np.int16).astype(int)

            i += 1
        else:
            output = [array[i]]
            if len(output) == k:
                return np.array([item[1] for item in output], dtype=np.int16).astype(int)
            i += 1


def show(img):
    cv2.imshow('Contours', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# array = np.array([5.414213538169861, 6.2426406145095825, 51.41421353816986, 68.24264061450958, 220.72792184352875, 227.65685415267944, 227.65685415267944, 759.019332408905, 759.019332408905, 760.1909114122391, 760.1909114122391, 818.7695519924164, 818.7695519924164, 818.7695519924164, 818.7695519924164, 818.7695519924164, 818.7695519924164, 818.7695519924164, 818.7695519924164, 818.7695519924164, 818.7695519924164, 818.7695519924164, 818.7695519924164, 818.7695519924164, 818.7695519924164, 818.7695519924164, 818.7695519924164, 818.7695519924164, 818.7695519924164, 818.7695519924164, 818.7695519924164, 818.7695519924164, 2509.9411249160767])
# np.random.shuffle(array)
# print(array)
# print(match_k_blobs(array=array, k=4, threshold=0.01))

