import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread("images/diaowu.png")
    for i in range(1000):
        print(i)
        img = cv2.medianBlur(img, 3)
        cv2.imshow("res", img)
        if cv2.waitKey(0) == 27:
            break
    cv2.destroyAllWindows()
