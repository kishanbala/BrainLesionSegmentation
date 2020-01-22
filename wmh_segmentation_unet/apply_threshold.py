from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

PATH = 'results\\35\\predict_out.png'

testImage = cv2.imread(PATH)
testArray = np.asarray(testImage, dtype=np.float32)
#testImageReverted = np.squeeze(testArray)

# plt.title('Actual')
# plt.imshow(testImageReverted)
# plt.show()

print(testArray.min(), testArray.max())

binaryArray = np.ndarray((np.shape(testArray)[0], np.shape(testArray)[1], np.shape(testArray)[2]), dtype=np.float32)

binaryArray[testArray >= 70] = 0
binaryArray[testArray < 70] = 1

binaryImage = np.squeeze(binaryArray)
plt.imsave('prediction.png', binaryImage)

#plt.title('Converted')
#plt.imshow(binaryImage)
#plt.show()