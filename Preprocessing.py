import cv2
import numpy as np

from keras_preprocessing.image import img_to_array
import functools
import matplotlib.pyplot as plt



#Main Function that return the Prediction number,the Path_image and Probabilities
def Preprocessing(model,image):
    thresh=read_image(image)
    mask=Border(image,thresh)
    BoundingBoxes=contour(mask)
    Numero,path,prob=prediction(BoundingBoxes,mask,image,model)
    return(Numero,path,prob)
#Read the image and return a blured image
def read_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 15)
    return(thresh)
#Detecting the borders
def Border(image,thresh) :
    none, labels = cv2.connectedComponents(thresh)
    mask = np.zeros(thresh.shape, dtype="uint8")

    for (i, label) in enumerate(np.unique(labels)):
        # If this is the background label, ignore it
        if label == 0:
            continue
        # Otherwise, construct the label mask to display only connected componen for the current label
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255

        numPixels = cv2.countNonZero(labelMask)
        # If the number of pixels in the component is over 100 pixels,add it to our mask
        if numPixels > 100:
            mask = cv2.add(mask, labelMask)
    return(mask)
#Detecting the contour of the image
def contour(mask):
        cnts, none = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        def compare(rect1, rect2):
            if abs(rect1[1] - rect2[1]) > 10:
                return rect1[1] - rect2[1]
            else:
                return rect1[0] - rect2[0]
        boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare) )
        return(boundingBoxes)
#The Prediction
def prediction(boundingBoxes,mask,image,model):
    Numero = ""
    probabilty=[]
    TARGET_WIDTH = 28
    TARGET_HEIGHT = 28
    # Loop over the bounding boxes
    for rect in boundingBoxes:
        # Get the coordinates from the bounding box
        x, y, w, h = rect
        # Crop the character from the mask
        # and apply bitwise_not to get black on a white background characters
        crop = mask[y - 10:y + h + 10, x - 10:x + w + 20]
        crop = cv2.bitwise_not(crop)
        # Get the number of rows and columns for each cropped image
        # and calculate the padding to match the image input of pre-trained model
        rows = crop.shape[0]
        columns = crop.shape[1]
        paddingY = (TARGET_HEIGHT - rows) // 2 if rows < TARGET_HEIGHT else int(0.17 * rows)
        paddingX = (TARGET_WIDTH - columns) // 2 if columns < TARGET_WIDTH else int(0.45 * columns)
        # Apply padding to make the image fit for neural network model
        crop = cv2.copyMakeBorder(crop, paddingY, paddingY, paddingX, paddingX, cv2.BORDER_CONSTANT, None, 255)
        # Convert and resize image
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        crop = cv2.resize(crop, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
        plt.imshow(crop)
        # Prepare data for prediction (Data Preprocessing)
        crop = crop / 255
        crop = img_to_array(crop)
        crop = np.expand_dims(crop, axis=0)
        gray_image = cv2.cvtColor(crop[0], cv2.COLOR_BGR2GRAY)
        gray_image = np.abs(gray_image - 1)
        gray_image = gray_image.reshape(1, 28, 28, 1)
        # Make prediction
        prob = model.predict(gray_image)[0]
        idx = np.argsort(prob)[-1]
        probabilty.append(np.max(prob))
        Numero += str(idx)
        # Show bounding box and prediction on image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, str(idx), (x, y + 15), 0, 0.8, (0, 0, 255), 2)
    # Show final image
    plt.imshow(image)
    path=f"result/{Numero}.png"
    #Save the image
    cv2.imwrite(path,image)
    plt.show()
    return(Numero,path,probabilty)

#Preprocessing(model,image)