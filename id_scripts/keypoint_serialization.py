import cv2
import numpy as np
import pickle

def __get_template_old_image_mask(width, height):
    ret = np.ones((height, width), np.uint8) * 255
    ret[:6, :] = 0
    ret[:40, 900:] = 0
    ret[:, 934:] = 0
    ret[:, :12] = 0
    ret[580:, :40] = 0
    ret[590:, :] = 0
    ret[560:, 920:] = 0
    ret[85:133, 310:653] = 0
    ret[155:228, 142:465] = 0
    ret[282:317,96:607] = 0
    ret[325:362,232:559] = 0
    ret[359:391,97:334] = 0
    ret[389:423,575:756] = 0
    ret[465:540,310:736] = 0
    return ret

def __get_template_old_image_mask_old(width, height):
    ret = np.ones((height, width), np.uint8) * 255
    ret[:6, :] = 0
    ret[:40, 900:] = 0
    ret[:, 934:] = 0
    ret[:, :12] = 0
    ret[580:, :40] = 0
    ret[590:, :] = 0
    ret[560:, 920:] = 0
    ret[87:133, 345:653] = 0
    ret[159:228, 152:465] = 0
    ret[282:315,96:601] = 0
    ret[325:362,232:559] = 0
    ret[389:423,595:756] = 0
    ret[465:537,310:736] = 0
    return ret


def __get_template_new_image_mask(width, height):
    ret = np.ones((height, width), np.uint8) * 255
    ret[:6, :] = 0
    ret[:36, :34] = 0
    ret[672:, :31] = 0
    ret[699:, :] = 0
    ret[658:, 908:] = 0
    ret[:, 951:] = 0
    ret[:39, 917:] = 0
    ret[:6, :] = 0
    ret[582:677,335:616] = 0
    ret[603:694,104:189] = 0
    ret[491:536,272:459] = 0
    ret[441:479,54:307] = 0
    ret[351:392,57:244] = 0
    ret[350:390,369:578] = 0
    ret[220:262,264:655] = 0
    ret[220:307,264:363] = 0
    ret[169:209,458:652] = 0
    ret[189:380,906:942] = 0
    return ret


if __name__ == "__main__":
    img1 = cv2.imread('/home/dani/Desktop/taxid_ocr/adokartyaocr/test_transformed_2.png') # queryImage
    sift = cv2.xfeatures2d.SIFT_create()
    mask = __get_template_old_image_mask(img1.shape[1], img1.shape[0])
    cv2.imshow("Masked", cv2.bitwise_and(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), mask))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    kp, des = sift.detectAndCompute(img1, mask)

    index = []
    for point in kp:
        temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
        index.append(temp)

    pickle.dump((index, des), open("data/template_old_camera.id_data", "wb"))

    (kp_array, des_copy) = pickle.load(open("data/template_old_camera.id_data", "rb"))

    kp_copy = []

    for point in index:
        temp = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        kp_copy.append(temp)

    print("Finished")