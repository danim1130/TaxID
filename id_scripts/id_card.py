import datetime
import pickle
import random
import logging
from collections import namedtuple
from enum import IntEnum


import cv2
import numpy as np
import pyzbar.pyzbar
import pylibdmtx.pylibdmtx

import id_scripts.pytesseract as pytesseract
import time
import re
from swagger_server.models.confidence_value import ConfidenceValue
from swagger_server.models.check_response_field import CheckResponseField
import logging
from main import application as app

Response = namedtuple('ID_Card', 'id_num name birthday birthplace mother_name_primary mother_name_secondary release_date type serial_number valid')


name_regex = re.compile('[^a-zA-ZáÁéÉíÍóÓöÖőŐüÜűŰúÚ.\- ]')
birthplace_regex = re.compile('[^a-zA-Z0-9áÁéÉíÍóÓöÖőŐüÜúÚ\- ]')
date_regex = re.compile('[^0-9.]')


def __run_tesseract_multiple_images(images, extension_configs, lang, run_otsu = False, blur_image = False, cluster_image_num = 0):
    start_time = time.time()
    for i in range(0, len(images)):
        image = images[i]
        if run_otsu:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if blur_image:
                image = cv2.GaussianBlur(image,(5,5),0.7)
            _, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
            #image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, (5, 5), iterations=1, borderValue=255)
            #image = cv2.morphologyEx(image, cv2.MORPH_DILATE, (5, 1), iterations=5, borderValue=255)
            image = cv2.copyMakeBorder(image, 25, 25, 15, 15, cv2.BORDER_CONSTANT, value=255)
        elif cluster_image_num != 0:
            if blur_image:
                image = cv2.GaussianBlur(image, (5,5), 0.7)

            Z = image.reshape((-1, 3))
            Z = np.float32(Z)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = cluster_image_num
            ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            center = np.uint8(center)
            minIndex = 0
            for j in range(1, K):
                if (sum(center[j]) < sum(center[minIndex])):
                    minIndex = j

            for j in range(0, K):
                if minIndex == j:
                    center[j][:] = 0
                else:
                    center[j][:] = 255

            res = center[label.flatten()]
            image = res.reshape((image.shape))
            image = cv2.copyMakeBorder(image, 25, 25, 15, 15, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        elif blur_image:
            image = cv2.GaussianBlur(image, (5,5), 0.7)

        images[i] = image

    start = time.time()
    read_str = pytesseract.run_multiple_and_get_output(images, extension='tsv', extension_configs=extension_configs, config="--psm 7", lang=lang)
    app.logger.log(logging.INFO, "Single tessract read run: {:.3f}".format(time.time() - start))
    #for idx, image in enumerate(images):
    #    cv2.imshow("TEST", image)
    #    cv2.waitKey(0)
        #cv2.imwrite("test%d.png" % idx, image, (cv2.IMWRITE_PNG_COMPRESSION, 0))

    read_values = []
    lines = read_str.split("\n")#
    for line in lines:
        read_values.append(line.split("\t"))

    page_index = read_values[0].index("page_num")
    base_row = read_values.pop(0)
    ret_list = [[]]
    ret_list[0].append(base_row)
    prev_page = '1'
    while len(read_values) != 0:
        next_row = read_values.pop(0)
        if next_row[page_index] != prev_page:
            ret_list.append([])
            ret_list[-1].append(base_row)
            prev_page = next_row[page_index]

        ret_list[-1].append(next_row)

    return ret_list
    #ret_list = []
    #with PyTessBaseAPI(lang=lang, psm=PSM.SINGLE_LINE) as api:
    #    for img in images:
    #        pil_im = Image.fromarray(img)
    #        api.SetImage(pil_im)
    #        ret_part = [["text","conf"]]
    #        texts_original = api.GetUTF8Text()
    #        texts = texts_original.split()
    #        confidences = api.AllWordConfidences()
    #        for i in range(0, len(texts)):
    #            ret_part.append([texts[i], confidences[i]])
    #        ret_list.append(ret_part)
    ##print("Tesseract time: " + str(time.time() - start_time))
    ##for idx, image in enumerate(images):
    #    #cv2.imshow("TEST", image)
    #    #cv2.waitKey(0)
    #    #cv2.imwrite("test%d.png" % idx, image, (cv2.IMWRITE_PNG_COMPRESSION, 0))
#
    #return ret_list


def __image_digits(read_values):
    date = ""
    confidence_level = 0
    text_index = read_values[0].index("text")
    confidence_index = read_values[0].index("conf")
    date_parts = []
    for i in range(1, len(read_values)):
        if len(read_values[i]) > text_index:
            date_parts_candidate = re.findall(r'\d+', read_values[i][text_index])
            #name_part_candidate = date_regex.sub('', read_values[i][text_index]).lstrip().rstrip()
            #date_parts_candidate = name_part_candidate.split(".")
            if 3 <= len(date_parts_candidate) <= 4 \
                    and len(date_parts_candidate[0]) == 4 and date_parts_candidate[0].isdigit() \
                    and len(date_parts_candidate[1]) <= 2 and date_parts_candidate[1].isdigit() \
                    and len(date_parts_candidate[2]) <= 2 and date_parts_candidate[2].isdigit() :
                date_parts = date_parts_candidate
                confidence_level = int(read_values[i][confidence_index])
                break

    if len(date_parts) == 0:
        return ConfidenceValue(value="", confidence=0)

    year = int(date_parts[0])
    if year < 1900:
        year = 1900 + (year % 100)
        confidence_level -= 10
    elif year > 2100:
        year = 2000 + (year % 100)
        confidence_level -= 10

    month = int(date_parts[1])
    if 12 < month < 20:
        month = 10
        confidence_level -= 10
    elif month > 21:
        month = (month % 10)
        confidence_level -= 10

    day = int(date_parts[2])
    date = str(year) + "." + str(month).zfill(2) + "." + str(day).zfill(2)
    return ConfidenceValue(value=date, confidence=confidence_level)


def __image_name(read_values):
    name = ""
    confidence_levels = []
    text_index = read_values[0].index("text")
    confidence_index = read_values[0].index("conf")
    for i in range(1, len(read_values)):
        if len(read_values[i]) > text_index:
            name_part_candidate = name_regex.sub('', read_values[i][text_index]).lstrip().rstrip()
            if len(name_part_candidate) != 0:
                confidence_levels.append(read_values[i][confidence_index])
                name = name + name_part_candidate + " "

    if len(name) == 0:
        return ConfidenceValue(value="", confidence=0), False

    #Fix name
    name = name.upper().rstrip().lstrip()

    name_parts = name.split(' ')
    for part in name_parts.copy()[::-1]:
        if len(part) == 0 or part[0] == '-' or part[0] == '.':
            confidence_levels.pop(name_parts.index(part))
            name_parts.remove(part)

    name_parts_temp = name_parts.copy()
    extra_fields = 0
    for part in name_parts[::-1]:
        if part == 'DR.' or part == "IFJ." or part == "ÖZV.":
            extra_fields += 1
            continue
        if part not in names:
            name_parts_temp.remove(part)
        else:
            break

    if len(name_parts_temp) > extra_fields + 1:
        return ConfidenceValue(value=' '.join(name_parts_temp).title(), confidence=min(confidence_levels[0:len(name_parts_temp) - 1])), True
    elif len(name_parts) == 0:
        return ConfidenceValue(value="", confidence=0), False
    else:
        return ConfidenceValue(value=' '.join(name_parts).title(), confidence=min(confidence_levels)), False


def roman_to_int(s):
    rom_val = {'|':1, 'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    int_val = 0
    for i in range(len(s)):
        if i > 0 and s[i] in rom_val and rom_val[s[i]] > rom_val[s[i - 1]]:
            int_val += rom_val[s[i]] - 2 * rom_val[s[i - 1]]
        else:
            int_val += rom_val[s[i]]
    return int_val


def __roman_match(strg, search=re.compile(r'[^|IVXLCDM]').search):
    return not bool(search(strg))


def __image_city(read_values):
    name = ""
    confidence_levels = []
    text_index = read_values[0].index("text")
    confidence_index = read_values[0].index("conf")
    for i in range(1, len(read_values)):
        if len(read_values[i]) > text_index:
            city_part_candidate = birthplace_regex.sub('', read_values[i][text_index]).lstrip().rstrip()
            if len(city_part_candidate) != 0:
                confidence_levels.append(read_values[i][confidence_index])
                name = name + city_part_candidate + " "

    if len(name) == 0:
        return ConfidenceValue(value="", confidence=0)

    #Fix city
    city = name.upper().rstrip().lstrip()#

    city_parts = city.split(' ')
    for part in city_parts.copy()[::-1]:
        if len(part) == 0 or part[0] == '-':
            confidence_levels.pop(city_parts.index(part))
            city_parts.remove(part)

    best_candidate = ''
    numbers_found = ''
    confidence_level = 100

    for part in city_parts[::-1]:
        if part in cities:
            best_candidate = part
            confidence_level = confidence_levels[city_parts.index(best_candidate)]
            break
        elif len(part) >= 2 and part[0:2].isdigit() and 0 < int(part[0:2]) <= 23:
            numbers_found = part[0:2]
        elif __roman_match(part):
            numbers_found = str(roman_to_int(part))
            if len(numbers_found) > 2:
                numbers_found = ""
        else:
            confidence_level = min(confidence_level, int(confidence_levels[city_parts.index(part)]))
            best_candidate = part + " " + best_candidate

    if len(numbers_found) != 0:
        best_candidate = best_candidate + " " + numbers_found

    return ConfidenceValue(value=best_candidate.title(), confidence=confidence_level)


def __check_field_match(input, field):
    return ConfidenceValue(value=input == field.value, confidence=int(field.confidence))



def __birth_date_from_id(num):  # https://hu.wikipedia.org/wiki/Ad%C3%B3azonos%C3%ADt%C3%B3_jel
    return datetime.date(1867, 1, 1) + datetime.timedelta(days=int(num[1:6]))


def __is_valid_id_num(num):  # https://hu.wikipedia.org/wiki/Ad%C3%B3azonos%C3%ADt%C3%B3_jel
    if len(num) != 10 or not num.isdigit():
        return False

    sum = 0
    for i in range(0, 9):
        sum += int(num[i]) * (i + 1)

    return sum % 11 == int(num[9])


with open('data/nevek.txt', 'r', encoding="utf-8") as f:
    names = frozenset(f.read().splitlines())

with open('data/varosok.txt', 'r', encoding="utf-8") as f:
    cities = frozenset(f.read().splitlines())


def __fix_name(name):
    name = name.upper()

    name_parts = name.split(' ')
    for part in name_parts.copy():
        if len(part) == 0 or part[0] == '-':
            name_parts.remove(part)

    name_parts_temp = name_parts.copy()
    for part in name_parts[::-1]:
        if part not in names:
            name_parts_temp.remove(part)
        else:
            break

    if len(name_parts_temp) > 1:
        return ' '.join(name_parts_temp).title()
    else:
        return ' '.join(name_parts).title()


def __fix_city(city):
    city = city.upper()

    city_parts = city.split(' ')
    for part in city_parts.copy():
        if len(part) <= 1:
            city_parts.remove(part)

    best_candidate = ''
    for part in city_parts:
        if part in cities:
            return part.title()
        else:
            best_candidate = best_candidate + " " + part

    return best_candidate.title()


(kp_array, old_card_des1) = pickle.load(open("data/template_old_camera.id_data", "rb"))
old_card_kp1 = []
for point in kp_array:
    temp = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
    old_card_kp1.append(temp)

(kp_array, old_card_des1_alt) = pickle.load(open("data/template_old_camera_old.id_data", "rb"))
old_card_kp1_alt = []
for point in kp_array:
    temp = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
    old_card_kp1_alt.append(temp)

(kp_array, new_card_des1) = pickle.load(open("data/template_new_camera.id_data", "rb"))
new_card_kp1 = []
for point in kp_array:
    temp = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
    new_card_kp1.append(temp)


class CardType(IntEnum):
    OLD_CARD = 0
    NEW_CARD = 1


card_sizes = [
    (940, 600), (960, 711)
]

field_coordinates = [
    [ #OLD_CARD
        [[285, 450], [780, 560]], #barcode
        [[140, 140], [720, 220]], #name
        [[385, 270], [710, 345]], #birthplace
        [[210, 320], [745, 370]], #mother_name_primary
        [[65, 355], [775, 405]], #mother_name_secondary
        [[565, 365], [770, 435]], #release data
    ],
    [ #NEW_CARD
        [[315, 573], [616, 675]], #barcode
        [[100, 600], [200, 700]], #datamatrix
        [[255, 210], [705, 270]], #name first#
        [[255, 260], [705, 310]], #name_second
        [[360, 343], [605, 400]], #birthplace
        [[45, 436], [545, 490]], #mother name
        [[263, 480], [469, 540]] #release date
    ]
]


def __get_image_part(img, card_type, card_index, override_coordinates = None):
    if override_coordinates is not None and len(override_coordinates) > card_type and len(override_coordinates[card_type]) > card_index and override_coordinates[card_type][card_index][0][0] != 0:
        return __get_image_part_by_coordinate(img, override_coordinates[card_type][card_index])
    else:
        return __get_image_part_by_coordinate(img, field_coordinates[card_type][card_index])


def __get_image_part_by_coordinate(img, coordinates):
    return img[coordinates[0][1]:coordinates[1][1], coordinates[0][0]:coordinates[1][0]]


def __get_transform_sift_for_type(input_img, card_type, target_width = 1280, runlevel = 0, use_alternate_card = False):
    MIN_MATCH_COUNT = 5

    img_height, img_width = input_img.shape[0:2]
    scale = (target_width / img_width)
    img2 = cv2.resize(input_img, (target_width, int(img_height * scale)), interpolation=cv2.INTER_LANCZOS4)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    if card_type == CardType.OLD_CARD:
        if use_alternate_card:
            kp1, des1 = old_card_kp1_alt, old_card_des1_alt
        else:
            kp1, des1 = old_card_kp1, old_card_des1
    elif card_type == CardType.NEW_CARD:
        kp1, des1 = new_card_kp1, new_card_des1

    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 100)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        if runlevel == 0:
            dst_pts /= scale
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            img2 = cv2.warpPerspective(input_img, M, card_sizes[card_type])
            #cv2.imshow("TEST", img2)
            #cv2.waitKey(0)
            #cv2.imwrite("test_transformed_fast.png", img2, (cv2.IMWRITE_PNG_COMPRESSION, 0))
            return img2

        offset = 40

        #M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)
        #cv2.imshow("TRANSFORMED", cv2.warpPerspective(img2, M, (card_sizes[card_type][0] + 2 * offset, card_sizes[card_type][1] + 2 * offset)))
        #cv2.waitKey(0)

        dst_pts /= scale
        src_pts += (offset, offset)

        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        img2 = cv2.warpPerspective(input_img, M, (card_sizes[card_type][0] + int(2 * offset), card_sizes[card_type][1] + int(2 * offset)))

        #img2 = cv2.fastNlMeansDenoisingColored(img2, None, 10, 10, 7, 21)

        kp2, des2 = sift.detectAndCompute(img2, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            if M is None:
                return "Transform not found"

            img2 = cv2.warpPerspective(img2, M, card_sizes[card_type])
            #cv2.imwrite("test_transformed.png", img2, (cv2.IMWRITE_PNG_COMPRESSION, 0))
            #cv2.imshow("TEST", img2)
            #cv2.waitKey(0)

            return img2
        else:
            return img2[offset:-offset,offset:-offset,:]
    else:
        return "Transform not found"

def __read_barcode_image(img):
    for i in range(1, 5):
        barcode_x_scale = 1 / i
        barcode_image = cv2.resize(img, (0, 0), fx=barcode_x_scale, fy=1)

        info = pyzbar.pyzbar.decode(barcode_image)
        if len(info) == 0:
            continue

        for barcode in info:
            if (barcode.type == "CODE128" or barcode.type == "I25") and __is_valid_id_num(barcode.data.decode("UTF-8")):
                return barcode.data.decode("UTF-8")


def __get_barcode_data(original_img):
    ret = __read_barcode_image(original_img)
    if ret is not None:
        return ret

    image = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0.7)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    img = cv2.copyMakeBorder(image, 25, 25, 15, 15, cv2.BORDER_CONSTANT, value=255)
    ret = __read_barcode_image(img)
    if ret is not None:
        return ret
    else:
        Z = original_img.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 5
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        minIndex = 0
        for j in range(1, K):
            if (sum(center[j]) < sum(center[minIndex])):
                minIndex = j

        for j in range(0, K):
            if minIndex == j:
                center[j][:] = 0
            else:
                center[j][:] = 255

        res = center[label.flatten()]
        img = res.reshape((original_img.shape))

        img = cv2.erode(img, np.ones((50,3),np.uint8))
        return __read_barcode_image(img)



def __get_datamatrix_data(img):
    for i in range(1, 5):
        barcode_x_scale = 1 / i
        barcode_image = cv2.resize(img, (0, 0), fx=barcode_x_scale, fy=1)

        info = pylibdmtx.pylibdmtx.decode(barcode_image)
        if len(info) == 0:
            continue
        else:
            break

    for barcode in info:
        return barcode.data.decode("UTF-8")



def __get_barcode_response(img):
    img_height, img_width = img.shape[0:2]
    img = cv2.resize(img, (640, int(img_height * (640 / img_width))), interpolation=cv2.INTER_CUBIC)
    temp_id_num = __get_barcode_data(img)
    if temp_id_num is None:
        return "ID card not found"

    real_birthday = ConfidenceValue(value=__birth_date_from_id(temp_id_num).strftime('%Y.%m.%d'), confidence=random.randint(90,100))
    return Response(id_num=temp_id_num,
                    name=ConfidenceValue("",0),
                    birthday=real_birthday,
                    birthplace=ConfidenceValue("",0),
                    release_date=ConfidenceValue("",0),
                    mother_name_primary=ConfidenceValue("",0),
                    mother_name_secondary=ConfidenceValue("",0),
                    serial_number=None,
                    type="UNKNOWN",
                    valid=__is_valid_id_num(temp_id_num))


def __add_to_list(dict, key, value):
    if key not in dict:
        dict[key] = [value]
    else:
        dict[key].append(value)


def __run_image_field_check(img, card_type, runlevel, unchecked_fields, validating_fields, found_fields, run_otsu, use_blur, cluster_image_num, override_coordinates):

    name = birthplace = mother_name_1 = mother_name_2 = release_date = serial_number = None

    if runlevel == 0:
        if card_type == CardType.OLD_CARD:
            image_parts = []
            if override_coordinates is None or override_coordinates[card_type][3][0][0] != 0:
                run_mother_1 = True
            else:
                run_mother_1 = False

            if override_coordinates is None or override_coordinates[card_type][4][0][0] != 0:
                run_mother_2 = True
            else:
                run_mother_2 = False

            if "name" in unchecked_fields:
                image_parts.append(__get_image_part(img, card_type,1, override_coordinates))
            if "birthplace" in unchecked_fields:
                image_parts.append(__get_image_part(img, card_type,2, override_coordinates))
            if "mother_name" in unchecked_fields:
                if run_mother_1:
                    image_parts.append(__get_image_part(img, card_type, 3, override_coordinates))
                if run_mother_2:
                    image_parts.append(__get_image_part(img, card_type, 4, override_coordinates))
            if "release_date" in unchecked_fields:
                image_parts.append(__get_image_part(img, card_type, 5, override_coordinates))

            if len(image_parts) != 0:
                tesseract_output = __run_tesseract_multiple_images(image_parts, extension_configs=["bazaar_complete"],
                                                                   lang="hun_fast", run_otsu=run_otsu, blur_image=use_blur, cluster_image_num=cluster_image_num)
            else:
                tesseract_output = []

            i = 0
            if "name" in unchecked_fields:
                name, found_name = __image_name(tesseract_output[i])
                i += 1
            if "birthplace" in unchecked_fields:
                birthplace = __image_city(tesseract_output[i])
                i += 1
            if "mother_name" in unchecked_fields:
                if run_mother_1:
                    mother_name_1, found_mother_1 = __image_name(tesseract_output[i])
                    i += 1
                if run_mother_2:
                    mother_name_2, found_mother_2 = __image_name(tesseract_output[i])
                    i += 1
            if "release_date" in unchecked_fields:
                release_date = __image_digits(tesseract_output[i])
            serial_number = None
        else:
            image_parts = []
            if "name" in unchecked_fields:
                image_parts.append(__get_image_part(img, card_type, 2, override_coordinates))
                image_parts.append(__get_image_part(img, card_type, 3, override_coordinates))
            if "birthplace" in unchecked_fields:
                image_parts.append(__get_image_part(img, card_type, 4, override_coordinates))
            if "mother_name" in unchecked_fields:
                image_parts.append(__get_image_part(img, card_type, 5, override_coordinates))
            if "release_date" in unchecked_fields:
                image_parts.append(__get_image_part(img, card_type, 6, override_coordinates))

            if len(image_parts) != 0:
                tesseract_output = __run_tesseract_multiple_images(image_parts, extension_configs=["bazaar_complete"],
                                                                   lang="hun_fast", run_otsu=run_otsu, blur_image=use_blur, cluster_image_num=cluster_image_num)
            else:
                tesseract_output = []

            i = 0
            if "name" in unchecked_fields:
                name, found_name = __image_name(tesseract_output[0] + tesseract_output[1][1:])
                i += 2
            if "birthplace" in unchecked_fields:
                birthplace = __image_city(tesseract_output[i])
                i += 1
            if "mother_name" in unchecked_fields:
                mother_name_1, found_mother_1 = __image_name(tesseract_output[i])
                mother_name_2, found_mother_2 = None, False
                i += 1
            if "release_date" in unchecked_fields:
                release_date = __image_digits(tesseract_output[i])
            if "serial_number" in unchecked_fields:
                serial_number = __get_datamatrix_data(__get_image_part(img, card_type, 1, override_coordinates))

    else:
        if card_type == CardType.OLD_CARD:
            image_parts = []
            if "name" in unchecked_fields:
                image_parts.append(__get_image_part(img, card_type, 1, override_coordinates))
            if "mother_name" in unchecked_fields:
                image_parts.append(__get_image_part(img, card_type, 3, override_coordinates))
                image_parts.append(__get_image_part(img, card_type, 4, override_coordinates))

            if len(image_parts) != 0:
                tesseract_output = __run_tesseract_multiple_images(image_parts, extension_configs=["bazaar_name"],
                                                                   lang="hun_fast", run_otsu=run_otsu, blur_image=use_blur, cluster_image_num=cluster_image_num)
            else:
                tesseract_output = []

            i = 0
            if "name" in unchecked_fields:
                name, found_name = __image_name(tesseract_output[0])
                i += 1
            if "mother_name" in unchecked_fields:
                mother_name_1, found_mother_1 = __image_name(tesseract_output[i])
                mother_name_2, found_mother_2 = __image_name(tesseract_output[i + 1])

            if "birthplace" in unchecked_fields:
                tesseract_output = __run_tesseract_multiple_images(
                    [__get_image_part(img, card_type, 2, override_coordinates)
                     ], extension_configs=["bazaar_city"], lang="hun_fast", run_otsu=run_otsu)
                birthplace = __image_city(tesseract_output[0])

            if "release_date" in unchecked_fields:
                tesseract_output = __run_tesseract_multiple_images(
                    [__get_image_part(img, card_type, 5, override_coordinates)
                     ], extension_configs=["digits"], lang="hun_fast", run_otsu=run_otsu)
                release_date = __image_digits(tesseract_output[0])

            serial_number = None
        else:
            image_parts = []
            if "name" in unchecked_fields:
                image_parts.append(__get_image_part(img, card_type, 2, override_coordinates))
                image_parts.append(__get_image_part(img, card_type, 3, override_coordinates))
            if "mother_name" in unchecked_fields:
                image_parts.append(__get_image_part(img, card_type, 5, override_coordinates))

            if len(image_parts) != 0:
                tesseract_output = __run_tesseract_multiple_images(image_parts, extension_configs=["bazaar_name"],
                                                                   lang="hun_fast", run_otsu=run_otsu, blur_image=use_blur, cluster_image_num=cluster_image_num)
            else:
                tesseract_output = []

            i = 0
            if "name" in unchecked_fields:
                name, found_name = __image_name(tesseract_output[0] + tesseract_output[1][1:])
                i += 2
            if "mother_name" in unchecked_fields:
                mother_name_1, found_mother_1 = __image_name(tesseract_output[i])
                mother_name_2, found_mother_2 = None, False

            if "birthplace" in unchecked_fields:
                tesseract_output = __run_tesseract_multiple_images(
                    [__get_image_part(img, card_type, 4, override_coordinates)
                     ], extension_configs=["bazaar_city"], lang="hun_fast", run_otsu=run_otsu)
                birthplace = __image_city(tesseract_output[0])

            if "release_date" in unchecked_fields:
                tesseract_output = __run_tesseract_multiple_images(
                    [__get_image_part(img, card_type, 6, override_coordinates)
                     ], extension_configs=["digits"], lang="hun_fast", run_otsu=run_otsu)
                release_date = __image_digits(tesseract_output[0])

            if "serial_number" in unchecked_fields:
                serial_number = __get_datamatrix_data(__get_image_part(img, card_type, 1, override_coordinates))

    if 'serial_number' in unchecked_fields:
        if serial_number != validating_fields['serial_number']:
            __add_to_list(found_fields, 'serial_number', serial_number)
        unchecked_fields.remove('serial_number')

    if 'name' in unchecked_fields and name is not None:
        if name.value == validating_fields['name']:
            unchecked_fields.remove('name')
        else:
            __add_to_list(found_fields, 'name', name)

    if 'birthplace' in unchecked_fields and birthplace is not None:
        if (birthplace.value == validating_fields['birthplace']):
            unchecked_fields.remove('birthplace')
        else:
            __add_to_list(found_fields, 'birthplace', birthplace)

    if 'mother_name' in unchecked_fields:
        if mother_name_1 is not None:
            if mother_name_1.value == validating_fields['mother_name']:
                unchecked_fields.remove('mother_name')
            else:
                __add_to_list(found_fields, 'mother_name', mother_name_1)
        if mother_name_2 is not None:
            if mother_name_2.value == validating_fields['mother_name']:
                unchecked_fields.remove('mother_name')
            else:
                __add_to_list(found_fields, 'mother_name', mother_name_2)

    if 'release_date' in unchecked_fields and release_date is not None:
        if validating_fields['release_date'][-1] == '.':
            validating_fields['release_date'] = validating_fields['release_date'][0:-1]
        if release_date.value == validating_fields['release_date']:
            unchecked_fields.remove('release_date')
        else:
            __add_to_list(found_fields, 'release_date', release_date)


text_coordinates = [
    [[0, 0]],
    [[215, 180], [215, 150], [150, 150], [150, 180], [230, 150], [230, 180]],
    [[430, 315], [430, 290], [460, 315], [460, 290]],
    [[240, 355], [300, 355], [350, 355]],
    [[100, 380], [130, 380], [100, 360], [130, 360]],
    [[600, 410], [640, 410], [600, 390], [640, 390]]
]


def __text_detect(image):
    ele_size = (25, 3)
    image_original = cv2.GaussianBlur(image, (3, 3), sigmaX=0.7)
    image = image_original[40:580, 40:850, :]
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 5
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    minIndex = 0
    for j in range(1, K):
        if (sum(center[j]) < sum(center[minIndex])):
            minIndex = j

    for j in range(0, K):
        if minIndex == j:
            center[j][:] = 0
        else:
            center[j][:] = 255

    res = center[label.flatten()]
    image = res.reshape((image.shape))

    if len(image.shape)==3:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.copyMakeBorder(image, 40, 0, 40, 0, cv2.BORDER_CONSTANT, value=255)
    img = cv2.Sobel(image,cv2.CV_8U,1,0)#same as default,None,3,1,0,cv2.BORDER_DEFAULT)
    img_threshold = cv2.threshold(img,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    element = cv2.getStructuringElement(cv2.MORPH_RECT,ele_size)
    img = cv2.morphologyEx(img_threshold[1],cv2.MORPH_CLOSE,element)
    #img_threshold = cv2.bitwise_not(img_threshold)
    contours = cv2.findContours(img,0,1)
    Rect_all = [cv2.boundingRect(i) for i in contours[1] if i.shape[0] > 40]
    Rect = [x for x in Rect_all if x[2] >= 40 and 20 <= x[3] <= 90]
    RectP = [(int(i[0]-i[2]*0.05),int(i[1]-i[3]*0.15),int(i[0]+i[2]*1.15),int(i[1]+i[3]*1.15)) for i in Rect]

    text_rects = [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)]
    for i in range(1, len(text_coordinates)):
        for j in range(len(text_coordinates[i])):
            found = False
            for rect in RectP:
                if rect[0] > text_coordinates[i][j][0] - 80 and rect[0] <= text_coordinates[i][j][0] <= rect[2] and \
                        rect[1] <= text_coordinates[i][j][1] <= rect[3]:
                    found = True
                    text_rects[i] = rect
                    break
            if found:
                break

    while True:
        changed = False
        for i in range(1, len(text_rects)):
            text_rect = text_rects[i]
            for rect in RectP:
                if rect[0] > text_rect[0] and rect[2] > text_rect[2] and rect[0] < text_rect[0] + 30 and (
                        (text_rect[1] - 5) <= rect[1] <= (text_rect[1]) + 5 or (text_rect[3]) - 5 <= rect[3] <= (
                        text_rect[3] + 5)):
                    text_rects[i] = (text_rect[0], min(rect[1], text_rect[1]), max(text_rect[2], rect[2]), max(text_rect[3], rect[3]))
                    changed = True

        if not changed:
            break

    #rect_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    #for i in RectP:
    #    cv2.rectangle(rect_image,i[:2],i[2:],(0,0,255))
    #for i in text_rects:
    #    cv2.rectangle(rect_image,i[:2],i[2:],(0,255,0))
    #for i in range(len(text_coordinates)):
    #    for j in range(len(text_coordinates[i])):
    #        cv2.circle(rect_image, (text_coordinates[i][j][0], text_coordinates[i][j][1]), 3, (255, 0, 0))
    #cv2.imshow("test", rect_image)
    #cv2.waitKey(0)

    for i in range(len(text_rects)):
        text_rects[i] = [[text_rects[i][0], text_rects[i][1]], [text_rects[i][2], text_rects[i][3]]]
    return [text_rects], image

def validate_id_card(img, runlevel, validating_fields):
    validating_fields = dict(validating_fields)
    unchecked_fields = set()

    for key in validating_fields:
        unchecked_fields.add(key)

    ###read_id_card
    start_time = time.time()

    original_img = img

    #if runlevel == 0:
    transform_target_width = 640
    #else:
    #    transform_target_width = 1280

    alternate_card_used = False
    img = __get_transform_sift_for_type(original_img, CardType.OLD_CARD, transform_target_width, 0)
    card_type = CardType.OLD_CARD
    if type(img) is str:
        alternate_card_used = True
        img = __get_transform_sift_for_type(original_img, CardType.OLD_CARD, transform_target_width, 0, use_alternate_card=True)
        if type(img) is str:
            card_type = CardType.NEW_CARD
            img = __get_transform_sift_for_type(original_img, CardType.NEW_CARD, transform_target_width, 0)
            if type(img) is str:
                return __get_barcode_response(original_img)

    app.logger.log(logging.INFO, "Warp time: {:.3f}".format(time.time() - start_time))
    # Barcode detection
    start = time.time()
    barcode_id_num = __get_barcode_data(__get_image_part(img, card_type, 0))
    if barcode_id_num is None:
        if card_type == CardType.OLD_CARD:
            if alternate_card_used:
                img = __get_transform_sift_for_type(original_img, CardType.NEW_CARD, transform_target_width, 0)
                card_type = CardType.NEW_CARD
                if type(img) is str:
                    return __get_barcode_response(original_img)
                else:
                    #cv2.imshow("Test", img)
                    #cv2.waitKey(0)
                    barcode_id_num = __get_barcode_data(__get_image_part(img, card_type, 0))
                    if barcode_id_num is None:
                        return __get_barcode_response(original_img)
            else:
                alternate_card_used = True
                img = __get_transform_sift_for_type(original_img, CardType.OLD_CARD, transform_target_width, 0, use_alternate_card=True)
                if type(img) is str:
                    img = __get_transform_sift_for_type(original_img, CardType.NEW_CARD, transform_target_width,0)
                    card_type = CardType.NEW_CARD
                    if type(img) is str:
                        return __get_barcode_response(original_img)
                    else:
                        # cv2.imshow("Test", img)
                        # cv2.waitKey(0)
                        barcode_id_num = __get_barcode_data(__get_image_part(img, card_type, 0))
                        if barcode_id_num is None:
                            return __get_barcode_response(original_img)
                    return __get_barcode_response(original_img)
                else:
                    barcode_id_num = __get_barcode_data(__get_image_part(img, card_type, 0))
                    if barcode_id_num is None:
                        img = __get_transform_sift_for_type(original_img, CardType.NEW_CARD, transform_target_width,
                                                            0)
                        card_type = CardType.NEW_CARD
                        if type(img) is str:
                            return __get_barcode_response(original_img)
                        else:
                            # cv2.imshow("Test", img)
                            # cv2.waitKey(0)
                            barcode_id_num = __get_barcode_data(__get_image_part(img, card_type, 0))
                            if barcode_id_num is None:
                                return __get_barcode_response(original_img)
        else:
            return __get_barcode_response(original_img)

    app.logger.log(logging.INFO, "Barcode time: {:.3f}".format(time.time() - start))

    real_id_num = barcode_id_num
    if (card_type == CardType.OLD_CARD):
        start = time.time()
        override_coordinates, clustered_image = __text_detect(img)
        app.logger.log(logging.INFO, "Text position detection time: {:.3f}".format(time.time() - start))
    else:
        override_coordinates, clustered_image = None, None


    real_birthday = ConfidenceValue(value=__birth_date_from_id(real_id_num).strftime('%Y.%m.%d'),
                                    confidence=random.randint(90, 100))

    if 'id_number' in unchecked_fields:
        unchecked_fields.remove('id_number')

    if 'birthdate' in unchecked_fields:
        unchecked_fields.remove('birthdate')
        if validating_fields['birthdate'][-1] == '.':
            validating_fields['birthdate'] = validating_fields['birthdate'][0:-1]

    found_fields = {}
    start = time.time()
    if clustered_image is not None and len(unchecked_fields) != 0:
        __run_image_field_check(clustered_image, card_type, runlevel, unchecked_fields, validating_fields, found_fields,
                                run_otsu=False, use_blur=False, cluster_image_num=0,
                                override_coordinates=override_coordinates)
    if len(unchecked_fields) != 0:  #79
        __run_image_field_check(img, card_type, runlevel, unchecked_fields, validating_fields, found_fields,
                                run_otsu=False, use_blur=True, cluster_image_num=5, override_coordinates=override_coordinates)
    if len(unchecked_fields) != 0: #79
        __run_image_field_check(img, card_type, runlevel, unchecked_fields, validating_fields, found_fields, run_otsu=True, use_blur=True, cluster_image_num=0, override_coordinates=override_coordinates)
    if len(unchecked_fields) != 0: #77
        __run_image_field_check(img, card_type, runlevel, unchecked_fields, validating_fields, found_fields, run_otsu=False, use_blur=True, cluster_image_num=0, override_coordinates=override_coordinates)
    if len(unchecked_fields) <= 3:
        if len(unchecked_fields) != 0: #75
            __run_image_field_check(img, card_type, runlevel, unchecked_fields, validating_fields, found_fields, run_otsu=True, use_blur=False, cluster_image_num=0, override_coordinates=override_coordinates)
        if len(unchecked_fields) != 0:  #71
            __run_image_field_check(img, card_type, runlevel, unchecked_fields, validating_fields, found_fields,
                                    run_otsu=False, use_blur=False, cluster_image_num=5, override_coordinates=override_coordinates)
    if len(unchecked_fields) <= 2:
        if len(unchecked_fields) != 0: #66
            __run_image_field_check(img, card_type, runlevel, unchecked_fields, validating_fields, found_fields, run_otsu=False, use_blur=False, cluster_image_num=6, override_coordinates=override_coordinates)
        if len(unchecked_fields) != 0:  # 65
            __run_image_field_check(img, card_type, runlevel, unchecked_fields, validating_fields, found_fields,
                                    run_otsu=False, use_blur=False, cluster_image_num=0, override_coordinates=override_coordinates)
        if card_type == CardType.OLD_CARD and alternate_card_used == False and len(unchecked_fields) != 0:
            img = __get_transform_sift_for_type(original_img, CardType.OLD_CARD, transform_target_width, 0, use_alternate_card=True)
            if type(img) is not str:
                override_coordinates, clustered_image = __text_detect(img)
                if clustered_image is not None and len(unchecked_fields) != 0:
                    __run_image_field_check(clustered_image, card_type, runlevel, unchecked_fields, validating_fields, found_fields,
                                            run_otsu=False, use_blur=False, cluster_image_num=0,
                                            override_coordinates=override_coordinates)
                if len(unchecked_fields) != 0: #63
                    __run_image_field_check(img, card_type, runlevel, unchecked_fields, validating_fields, found_fields,
                                            run_otsu=False, use_blur=True, cluster_image_num=5, override_coordinates=override_coordinates)
                if len(unchecked_fields) != 0: #58
                    __run_image_field_check(img, card_type, runlevel, unchecked_fields, validating_fields, found_fields,
                                            run_otsu=False, use_blur=False, cluster_image_num=5, override_coordinates=override_coordinates)
                if len(unchecked_fields) != 0: #58
                    __run_image_field_check(img, card_type, runlevel, unchecked_fields, validating_fields, found_fields,
                                            run_otsu=True, use_blur=True, cluster_image_num=0, override_coordinates=override_coordinates)
                if len(unchecked_fields) != 0: #56
                    __run_image_field_check(img, card_type, runlevel, unchecked_fields, validating_fields, found_fields,
                                            run_otsu=False, use_blur=True, cluster_image_num=0, override_coordinates=override_coordinates)

                if len(unchecked_fields) != 0: #54
                    __run_image_field_check(img, card_type, runlevel, unchecked_fields, validating_fields, found_fields,
                                            run_otsu=False, use_blur=False, cluster_image_num=6, override_coordinates=override_coordinates)
                if len(unchecked_fields) != 0: #51
                    __run_image_field_check(img, card_type, runlevel, unchecked_fields, validating_fields, found_fields,
                                            run_otsu=True, use_blur=False, cluster_image_num=0, override_coordinates=override_coordinates)
                if len(unchecked_fields) != 0: #48
                    __run_image_field_check(img, card_type, runlevel, unchecked_fields, validating_fields, found_fields,
                                        run_otsu=False, use_blur=False, cluster_image_num=0, override_coordinates=override_coordinates)

    app.logger.log(logging.INFO, "Text read complete time: {:.3f}".format(time.time() - start))

    valid_failed = False
    valid_success = True

    start = time.time()
    response = {}
    if 'id_number' in validating_fields:
        if real_id_num == validating_fields['id_number']:
            response['id_number'] = CheckResponseField(validation_result=ConfidenceValue(True, 100))
        else:
            valid_failed = True
            valid_success = False
            response['id_number'] = CheckResponseField(validation_result=ConfidenceValue(value=False, confidence=100),
                                                       possible_values=[ConfidenceValue(real_id_num, confidence=100)])
        del validating_fields['id_number']

    if 'birthdate' in validating_fields:
        if real_birthday is not None and real_birthday.value == validating_fields['birthdate']:
            response['birthdate'] = CheckResponseField(validation_result=ConfidenceValue(True, 100))
        else:
            valid_failed = True
            valid_success = False
            response['birthdate'] = CheckResponseField(validation_result=ConfidenceValue(value=False, confidence=100),
                                                       possible_values=[real_birthday])
        del validating_fields['birthdate']

    for key in validating_fields:
        if key not in unchecked_fields:
            response[key] = CheckResponseField(validation_result=ConfidenceValue(True, 100))
        elif key in found_fields and found_fields[key] is not None:
            maxValue = -1
            maxField = ""

            field_parts = validating_fields[key].split(' ')
            for conf in found_fields[key]:
                field_parts[:] = [part for part in field_parts if (part not in conf.value.split(' '))]
                if int(conf.confidence) > maxValue and conf.value is not None and type(conf.value) is str and len(conf.value) != 0:
                    maxValue = int(conf.confidence)
                    maxField = conf

            if len(field_parts) == 0:
                response[key] = CheckResponseField(validation_result=ConfidenceValue(True, 100))
            elif maxValue == -1:
                response[key] = CheckResponseField(validation_result=ConfidenceValue(False, maxValue))
                valid_success = False
            else:
                response[key] = CheckResponseField(possible_values=[maxField], validation_result=ConfidenceValue(False, maxValue))
                valid_success = False
        else:
            response[key] = CheckResponseField(validation_result=ConfidenceValue(value=False, confidence=0))

    response['validation_success'] = valid_success
    response['validation_failed'] = valid_failed
    app.logger.log(logging.INFO, "Post processing time: {:.3f}".format(time.time() - start))
    app.logger.log(logging.INFO, "Complete processing time: {:.3f}".format(time.time() - start_time))

    return response


#testresult = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#
#
#def validate_id_card_with_test(img, runlevel, validating_fields):
#    validating_fields = dict(validating_fields)
#    unchecked_fields = set()
#
#    for key in validating_fields:
#        unchecked_fields.add(key)
#
#    ###read_id_card
#    start_time = time.time()
#
#    original_img = img
#
#    #if runlevel == 0:
#    transform_target_width = 640
#    #else:
#    #    transform_target_width = 1280
#
#    alternate_card_used = False
#    img = __get_transform_sift_for_type(original_img, CardType.OLD_CARD, transform_target_width, 0)
#    card_type = CardType.OLD_CARD
#    if type(img) is str:
#        alternate_card_used = True
#        img = __get_transform_sift_for_type(original_img, CardType.OLD_CARD, transform_target_width, 0, use_alternate_card=True)
#        if type(img) is str:
#            card_type = CardType.NEW_CARD
#            img = __get_transform_sift_for_type(original_img, CardType.NEW_CARD, transform_target_width, 0)
#            if type(img) is str:
#                return __get_barcode_response(original_img)
#
#    #print("Warp time: " + str((time.time() - start_time)))
#
#    # Barcode detection
#    barcode_id_num = __get_barcode_data(__get_image_part(img, field_coordinates[card_type][0]))
#    if barcode_id_num is None:
#        if card_type == CardType.OLD_CARD:
#            if alternate_card_used:
#                img = __get_transform_sift_for_type(original_img, CardType.NEW_CARD, transform_target_width, 0)
#                card_type = CardType.NEW_CARD
#                if type(img) is str:
#                    return __get_barcode_response(original_img)
#                else:
#                    #cv2.imshow("Test", img)
#                    #cv2.waitKey(0)
#                    barcode_id_num = __get_barcode_data(__get_image_part(img, field_coordinates[card_type][0]))
#                    if barcode_id_num is None:
#                        return __get_barcode_response(original_img)
#            else:
#                alternate_card_used = True
#                img = __get_transform_sift_for_type(original_img, CardType.NEW_CARD, transform_target_width, 0, use_alternate_card=True)
#                if type(img) is str:
#                    img = __get_transform_sift_for_type(original_img, CardType.NEW_CARD, transform_target_width,0)
#                    card_type = CardType.NEW_CARD
#                    if type(img) is str:
#                        return __get_barcode_response(original_img)
#                    else:
#                        # cv2.imshow("Test", img)
#                        # cv2.waitKey(0)
#                        barcode_id_num = __get_barcode_data(__get_image_part(img, field_coordinates[card_type][0]))
#                        if barcode_id_num is None:
#                            return __get_barcode_response(original_img)
#                    return __get_barcode_response(original_img)
#                else:
#                    barcode_id_num = __get_barcode_data(__get_image_part(img, field_coordinates[card_type][0]))
#                    if barcode_id_num is None:
#                        img = __get_transform_sift_for_type(original_img, CardType.NEW_CARD, transform_target_width,
#                                                            0)
#                        card_type = CardType.NEW_CARD
#                        if type(img) is str:
#                            return __get_barcode_response(original_img)
#                        else:
#                            # cv2.imshow("Test", img)
#                            # cv2.waitKey(0)
#                            barcode_id_num = __get_barcode_data(__get_image_part(img, field_coordinates[card_type][0]))
#                            if barcode_id_num is None:
#                                return __get_barcode_response(original_img)
#        else:
#            return __get_barcode_response(original_img)
#
#    real_id_num = barcode_id_num
#    #print("Barcode time: " + str((time.time() - start_time)))
#
#    real_birthday = ConfidenceValue(value=__birth_date_from_id(real_id_num).strftime('%Y.%m.%d'),
#                                    confidence=random.randint(90, 100))
#
#    if 'id_number' in unchecked_fields:
#        unchecked_fields.remove('id_number')
#
#    if 'birthdate' in unchecked_fields:
#        unchecked_fields.remove('birthdate')
#        if validating_fields['birthdate'][-1] == '.':
#            validating_fields['birthdate'] = validating_fields['birthdate'][0:-1]
#
#    found_fields = {}
#    unchecked_copy = unchecked_fields.copy()
#    __run_image_field_check(img, card_type, runlevel, unchecked_copy, validating_fields, found_fields, run_otsu=False, use_blur=False, cluster_image_num=5)
#    testresult[0] += len(unchecked_fields) - len(unchecked_copy)
#    unchecked_copy = unchecked_fields.copy()
#    __run_image_field_check(img, card_type, runlevel, unchecked_copy, validating_fields, found_fields, run_otsu=False, use_blur=True, cluster_image_num=5)
#    testresult[1] += len(unchecked_fields) - len(unchecked_copy)
#    unchecked_copy = unchecked_fields.copy()
#    __run_image_field_check(img, card_type, runlevel, unchecked_copy, validating_fields, found_fields, run_otsu=False, use_blur=False, cluster_image_num=0)
#    testresult[2] += len(unchecked_fields) - len(unchecked_copy)
#    unchecked_copy = unchecked_fields.copy()
#    __run_image_field_check(img, card_type, runlevel, unchecked_copy, validating_fields, found_fields, run_otsu=True, use_blur=False, cluster_image_num=0)
#    testresult[3] += len(unchecked_fields) - len(unchecked_copy)
#    unchecked_copy = unchecked_fields.copy()
#    __run_image_field_check(img, card_type, runlevel, unchecked_copy, validating_fields, found_fields, run_otsu=False, use_blur=True, cluster_image_num=0)
#    testresult[4] += len(unchecked_fields) - len(unchecked_copy)
#    unchecked_copy = unchecked_fields.copy()
#
#    __run_image_field_check(img, card_type, runlevel, unchecked_copy, validating_fields, found_fields, run_otsu=False, use_blur=False, cluster_image_num=6)
#    testresult[5] += len(unchecked_fields) - len(unchecked_copy)
#    unchecked_copy = unchecked_fields.copy()
#    __run_image_field_check(img, card_type, runlevel, unchecked_copy, validating_fields, found_fields, run_otsu=True, use_blur=True, cluster_image_num=0)
#    testresult[6] += len(unchecked_fields) - len(unchecked_copy)
#    unchecked_copy = unchecked_fields.copy()
#
#    img = __get_transform_sift_for_type(original_img, CardType.OLD_CARD, transform_target_width, 0, use_alternate_card=True)
#    __run_image_field_check(img, card_type, runlevel, unchecked_copy, validating_fields, found_fields,
#                            run_otsu=False, use_blur=False, cluster_image_num=5)
#    testresult[7] += len(unchecked_fields) - len(unchecked_copy)
#    unchecked_copy = unchecked_fields.copy()
#    __run_image_field_check(img, card_type, runlevel, unchecked_copy, validating_fields, found_fields,
#                            run_otsu=False, use_blur=False, cluster_image_num=6)
#    testresult[8] += len(unchecked_fields) - len(unchecked_copy)
#    unchecked_copy = unchecked_fields.copy()
#    __run_image_field_check(img, card_type, runlevel, unchecked_copy, validating_fields, found_fields,
#                            run_otsu=False, use_blur=True, cluster_image_num=5)
#    testresult[9] += len(unchecked_fields) - len(unchecked_copy)
#    unchecked_copy = unchecked_fields.copy()
#    __run_image_field_check(img, card_type, runlevel, unchecked_copy, validating_fields, found_fields,
#                        run_otsu=False, use_blur=False, cluster_image_num=0)
#    testresult[10] += len(unchecked_fields) - len(unchecked_copy)
#    unchecked_copy = unchecked_fields.copy()
#    __run_image_field_check(img, card_type, runlevel, unchecked_copy, validating_fields, found_fields,
#                            run_otsu=True, use_blur=False, cluster_image_num=0)
#    testresult[11] += len(unchecked_fields) - len(unchecked_copy)
#    unchecked_copy = unchecked_fields.copy()
#    __run_image_field_check(img, card_type, runlevel, unchecked_copy, validating_fields, found_fields,
#                            run_otsu=True, use_blur=True, cluster_image_num=0)
#    testresult[12] += len(unchecked_fields) - len(unchecked_copy)
#    unchecked_copy = unchecked_fields.copy()
#    __run_image_field_check(img, card_type, runlevel, unchecked_copy, validating_fields, found_fields,
#                            run_otsu=False, use_blur=True, cluster_image_num=0)
#    testresult[13] += len(unchecked_fields) - len(unchecked_copy)
#
#    with open("result.txt", "w") as file:
#        file.write(str(testresult))
#
#
#    valid_failed = False
#    valid_success = True
#
#    response = {}
#    if 'id_number' in validating_fields:
#        if real_id_num == validating_fields['id_number']:
#            response['id_number'] = CheckResponseField(validation_result=ConfidenceValue(True, 100))
#        else:
#            valid_failed = True
#            valid_success = False
#            response['id_number'] = CheckResponseField(validation_result=ConfidenceValue(value=False, confidence=100),
#                                                       possible_values=[ConfidenceValue(real_id_num, confidence=100)])
#        del validating_fields['id_number']
#
#    if 'birthdate' in validating_fields:
#        if real_birthday is not None and real_birthday.value == validating_fields['birthdate']:
#            response['birthdate'] = CheckResponseField(validation_result=ConfidenceValue(True, 100))
#        else:
#            valid_failed = True
#            valid_success = False
#            response['birthdate'] = CheckResponseField(validation_result=ConfidenceValue(value=False, confidence=100),
#                                                       possible_values=[real_birthday])
#        del validating_fields['birthdate']
#
#    for key in validating_fields:
#        if key not in unchecked_fields:
#            response[key] = CheckResponseField(validation_result=ConfidenceValue(True, 100))
#        elif found_fields[key] is not None:
#            maxValue = -1
#            maxField = ""
#
#            field_parts = validating_fields[key].split(' ')
#            for conf in found_fields[key]:
#                field_parts[:] = [part for part in field_parts if (part not in conf.value.split(' '))]
#                if int(conf.confidence) > maxValue and conf.value is not None and type(conf.value) is str and len(conf.value) != 0:
#                    maxValue = int(conf.confidence)
#                    maxField = conf
#
#            if len(field_parts) == 0:
#                response[key] = CheckResponseField(validation_result=ConfidenceValue(True, 100))
#            elif maxValue == -1:
#                response[key] = CheckResponseField(validation_result=ConfidenceValue(False, maxValue))
#                valid_success = False
#            else:
#                response[key] = CheckResponseField(possible_values=[maxField], validation_result=ConfidenceValue(False, maxValue))
#                valid_success = False
#        else:
#            response[key] = CheckResponseField(validation_result=ConfidenceValue(value=False, confidence=0))
#
#    response['validation_success'] = valid_success
#    response['validation_failed'] = valid_failed
#    return response



def read_id_card(img, runlevel, filter_fields = [], run_otsu = False, use_blur = False):
    start_time = time.time()

    original_img = img

    if runlevel == 0:
        transform_target_width = 640
    else:
        transform_target_width = 1280

    img = __get_transform_sift_for_type(original_img, CardType.OLD_CARD, transform_target_width, runlevel)
    card_type = CardType.OLD_CARD
    if type(img) is str:
        card_type = CardType.NEW_CARD
        img = __get_transform_sift_for_type(original_img, CardType.NEW_CARD, transform_target_width, runlevel)
        if type(img) is str:
            return __get_barcode_response(original_img)

    alternate_card_used = False
    img = __get_transform_sift_for_type(original_img, CardType.OLD_CARD, transform_target_width, runlevel)
    card_type = CardType.OLD_CARD
    if type(img) is str:
        alternate_card_used = True
        img = __get_transform_sift_for_type(original_img, CardType.OLD_CARD, transform_target_width, runlevel, use_alternate_card=True)
        if type(img) is str:
            card_type = CardType.NEW_CARD
            img = __get_transform_sift_for_type(original_img, CardType.NEW_CARD, transform_target_width, runlevel)
            if type(img) is str:
                return __get_barcode_response(original_img)

    # Barcode detection
    barcode_id_num = __get_barcode_data(__get_image_part(img, card_type, 0, override_coordinates))
    if barcode_id_num is None:
        if card_type == CardType.OLD_CARD:
            if alternate_card_used:
                img = __get_transform_sift_for_type(original_img, CardType.NEW_CARD, transform_target_width, runlevel)
                card_type = CardType.NEW_CARD
                if type(img) is str:
                    return __get_barcode_response(original_img)
                else:
                    #cv2.imshow("Test", img)
                    #cv2.waitKey(0)
                    barcode_id_num = __get_barcode_data(__get_image_part(img, card_type, 0, override_coordinates))
                    if barcode_id_num is None:
                        return __get_barcode_response(original_img)
            else:
                alternate_card_used = True
                img = __get_transform_sift_for_type(original_img, CardType.NEW_CARD, transform_target_width, runlevel, use_alternate_card=True)
                if type(img) is str:
                    img = __get_transform_sift_for_type(original_img, CardType.NEW_CARD, transform_target_width,runlevel)
                    card_type = CardType.NEW_CARD
                    if type(img) is str:
                        return __get_barcode_response(original_img)
                    else:
                        #cv2.imshow("Test", img)
                        #cv2.waitKey(0)
                        barcode_id_num = __get_barcode_data(__get_image_part(img, card_type, 0, override_coordinates))
                        if barcode_id_num is None:
                            return __get_barcode_response(original_img)
                    return __get_barcode_response(original_img)
                else:
                    barcode_id_num = __get_barcode_data(__get_image_part(img, card_type, 0, override_coordinates))
                    if barcode_id_num is None:
                        img = __get_transform_sift_for_type(original_img, CardType.NEW_CARD, transform_target_width,
                                                            runlevel)
                        card_type = CardType.NEW_CARD
                        if type(img) is str:
                            return __get_barcode_response(original_img)
                        else:
                            # cv2.imshow("Test", img)
                            # cv2.waitKey(0)
                            barcode_id_num = __get_barcode_data(__get_image_part(img, card_type, 0, override_coordinates))
                            if barcode_id_num is None:
                                return __get_barcode_response(original_img)
        else:
            return __get_barcode_response(original_img)

    real_id_num = barcode_id_num
    #print("Barcode time: " + str((time.time() - start_time)))

    real_birthday = ConfidenceValue(value=__birth_date_from_id(real_id_num).strftime('%Y.%m.%d'), confidence=random.randint(90,100))

    start_time = time.time()

    #Text areas
    #imgCopy = img.copy()
    #for rect in field_coordinates[card_type]:
    #    cv2.rectangle(imgCopy, (rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]), 0)
    #cv2.imshow("Test2", imgCopy)
    #cv2.waitKey(0)
    ##cv2.imwrite("test_output/test.jpg", imgCopy)
    #for i in range(1, 7):
    #    cv2.imwrite("test_output/test" + str(i) + ".jpg", __get_image_part(img, card_type, i, override_coordinates))

    name = birthplace = mother_name_1 = mother_name_2 = release_date = serial_number = None
    found_name = found_mother_1 = found_mother_2 = False

    if runlevel == 0 or runlevel == 1:
        if card_type == CardType.OLD_CARD:
            image_parts = []
            if "name" not in filter_fields:
                image_parts.append(__get_image_part(img, card_type, 1, override_coordinates))
            if "birthplace" not in filter_fields:
                image_parts.append(__get_image_part(img, card_type, 2, override_coordinates))
            if "mother_name" not in filter_fields:
                image_parts.append(__get_image_part(img, card_type, 3, override_coordinates))
                image_parts.append(__get_image_part(img, card_type, 4, override_coordinates))
            if "release_date" not in filter_fields:
                image_parts.append(__get_image_part(img, card_type, 5, override_coordinates))

            if len(image_parts) != 0:
                tesseract_output = __run_tesseract_multiple_images(image_parts, extension_configs=["bazaar_complete"], lang="hun_fast", run_otsu=run_otsu, blur_image=use_blur)
            else:
                tesseract_output = []

            i = 0
            if "name" not in filter_fields:
                name, found_name = __image_name(tesseract_output[i])
                i += 1
            if "birthplace" not in filter_fields:
                birthplace = __image_city(tesseract_output[i])
                i += 1
            if "mother_name" not in filter_fields:
                mother_name_1, found_mother_1 = __image_name(tesseract_output[i])
                i += 1
                mother_name_2, found_mother_2 = __image_name(tesseract_output[i])
                i += 1
            if "release_date" not in filter_fields:
                release_date = __image_digits(tesseract_output[i])
            serial_number = None
        else:
            image_parts = []
            if "name" not in filter_fields:
                image_parts.append(__get_image_part(img, card_type, 2, override_coordinates))
                image_parts.append(__get_image_part(img, card_type, 3, override_coordinates))
            if "birthplace" not in filter_fields:
                image_parts.append(__get_image_part(img, card_type, 4, override_coordinates))
            if "mother_name" not in filter_fields:
                image_parts.append(__get_image_part(img, card_type, 5, override_coordinates))
            if "release_date" not in filter_fields:
                image_parts.append(__get_image_part(img, card_type, 6, override_coordinates))

            if len(image_parts) != 0:
                tesseract_output = __run_tesseract_multiple_images(image_parts, extension_configs=["bazaar_complete"], lang="hun_fast", run_otsu=run_otsu, blur_image=use_blur)
            else:
                tesseract_output = []

            i = 0
            if "name" not in filter_fields:
                name, found_name = __image_name(tesseract_output[0] + tesseract_output[1][1:])
                i += 2
            if "birthplace" not in filter_fields:
                birthplace = __image_city(tesseract_output[i])
                i += 1
            if "mother_name" not in filter_fields:
                mother_name_1, found_mother_1 = __image_name(tesseract_output[i])
                mother_name_2, found_mother_2 = None, False
                i += 1
            if "release_date" not in filter_fields:
                release_date = __image_digits(tesseract_output[i])
            if "serial_number" not in filter_fields:
                serial_number = __get_datamatrix_data(__get_image_part(img, card_type, 1, override_coordinates))

    else:
        if card_type == CardType.OLD_CARD:
            image_parts = []
            if "name" not in filter_fields:
                image_parts.append(__get_image_part(img, card_type, 1, override_coordinates))
            if "mother_name" not in filter_fields:
                image_parts.append(__get_image_part(img, card_type, 3, override_coordinates))
                image_parts.append(__get_image_part(img, card_type, 4, override_coordinates))

            if len(image_parts) != 0:
                tesseract_output = __run_tesseract_multiple_images(image_parts, extension_configs=["bazaar_name"], lang="hun_fast", run_otsu=run_otsu, blur_image=use_blur)
            else:
                tesseract_output = []

            i = 0
            if "name" not in filter_fields:
                name, found_name = __image_name(tesseract_output[0])
                i += 1
            if "mother_name" not in filter_fields:
                mother_name_1, found_mother_1 = __image_name(tesseract_output[i])
                mother_name_2, found_mother_2 = __image_name(tesseract_output[i+1])

            if "birthplace" not in filter_fields:
                tesseract_output = __run_tesseract_multiple_images([__get_image_part(img, card_type, 2, override_coordinates)
                                                                    ], extension_configs=["bazaar_city"], lang="hun_fast", run_otsu=run_otsu, blur_image=use_blur)
                birthplace = __image_city(tesseract_output[0])

            if "release_date" not in filter_fields:
                tesseract_output = __run_tesseract_multiple_images([__get_image_part(img, card_type, 5, override_coordinates)
                                                                    ], extension_configs=["digits"], lang="hun_fast", run_otsu=run_otsu, blur_image=use_blur)
                release_date = __image_digits(tesseract_output[0])

            serial_number = None
        else:
            image_parts = []
            if "name" not in filter_fields:
                image_parts.append(__get_image_part(img, card_type, 2, override_coordinates))
                image_parts.append(__get_image_part(img, card_type, 3, override_coordinates))
            if "mother_name" not in filter_fields:
                image_parts.append(__get_image_part(img, card_type, 5, override_coordinates))

            if len(image_parts) != 0:
                tesseract_output = __run_tesseract_multiple_images(image_parts, extension_configs=["bazaar_name"], lang="hun_fast", run_otsu=run_otsu, blur_image=use_blur)
            else:
                tesseract_output = []

            i = 0
            if "name" not in filter_fields:
                name, found_name = __image_name(tesseract_output[0] + tesseract_output[1][1:])
                i += 2
            if "mother_name" not in filter_fields:
                mother_name_1, found_mother_1 = __image_name(tesseract_output[i])
                mother_name_2, found_mother_2 = None, False

            if "birthplace" not in filter_fields:
                tesseract_output = __run_tesseract_multiple_images([__get_image_part(img, card_type, 4, override_coordinates)
                                                                    ], extension_configs=["bazaar_city"], lang="hun_fast", run_otsu=run_otsu, blur_image=use_blur)
                birthplace = __image_city(tesseract_output[0])

            if "release_date" not in filter_fields:
                tesseract_output = __run_tesseract_multiple_images([__get_image_part(img, card_type, 6, override_coordinates)
                                                                    ], extension_configs=["digits"], lang="hun_fast", run_otsu=run_otsu, blur_image=use_blur)
                release_date = __image_digits(tesseract_output[0])

            if "serial_number" not in filter_fields:
                serial_number = __get_datamatrix_data(__get_image_part(img, card_type, 1, override_coordinates))


    #print("Read time: " + str((time.time() - start_time)))

    if mother_name_2 is None:
        mother_name_primary = mother_name_1
        mother_name_secondary = mother_name_2
    elif found_mother_1:
        mother_name_primary = mother_name_1
        mother_name_secondary = mother_name_2
    elif found_mother_2:
        mother_name_primary = mother_name_2
        mother_name_secondary = mother_name_1
    elif len(mother_name_1.value.split()) <= 3 <= len(mother_name_2.value.split()):
        mother_name_primary = mother_name_1
        mother_name_secondary = mother_name_2
    elif len(mother_name_1.value.split()) >= 3 >= len(mother_name_2.value.split()):
        mother_name_primary = mother_name_2
        mother_name_secondary = mother_name_1
    elif len(mother_name_1.value) > len(mother_name_2.value):
        mother_name_primary = mother_name_1
        mother_name_secondary = mother_name_2
    elif len(mother_name_2.value) > len(mother_name_1.value):
        mother_name_primary = mother_name_2
        mother_name_secondary = mother_name_1
    else:
        mother_name_primary = mother_name_1
        mother_name_secondary = mother_name_2

    return Response(id_num=real_id_num,
                    name=name,
                    birthday=real_birthday,
                    birthplace=birthplace,
                    release_date=release_date,
                    mother_name_primary=mother_name_primary,
                    mother_name_secondary=mother_name_secondary,
                    valid=__is_valid_id_num(real_id_num),
                    serial_number=serial_number,
                    type=card_type.name)
