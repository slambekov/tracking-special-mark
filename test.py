import cv2
import os
execution_path = os.getcwd()
input_file_path=os.path.join(execution_path, "icon.png"),
s_img = cv2.imread("icon.png")
s_img = cv2.resize(s_img,(4,4),cv2.INTER_AREA)
print(s_img)
dimensions = s_img.shape
test = s_img.reshape(-1,4)
print(test)
print(test.shape)