import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
# img = cv2.imread('OpenCV_Logo_B.png')
# mask = cv2.imread('OpenCV_Logo_C.png', 0)

# path = os.path.join('D:\\S\\记录\\工作\\SLAM\\RGBD', '2021-10-26 17-25-20屏幕截图.png')
img = cv2.imread('woman-ble.png')
img = img[:,:,[2,1,0]]
mask = cv2.imread('woman-ble-manualmask.png', 0)

dst_TELEA = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
dst_NS = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)

plt.imshow(img)
plt.figure()
plt.imshow(dst_TELEA)
plt.figure()
plt.imshow(dst_NS)
plt.show()