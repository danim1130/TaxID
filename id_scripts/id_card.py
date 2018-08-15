import datetime
import pickle
import random
import re
from collections import namedtuple
from enum import IntEnum

import cv2
import numpy as np
import pyzbar.pyzbar
import pylibdmtx.pylibdmtx

import id_scripts.pytesseract as pytesseract
import time
from swagger_server.models.confidence_value import ConfidenceValue
from swagger_server.models.check_response_field import CheckResponseField

Response = namedtuple('ID_Card', 'id_num name birthday birthplace mother_name_primary mother_name_secondary release_date type serial_number valid')


name_regex = re.compile('[^a-zA-ZáÁéÉíÍóÓöÖőŐüÜúÚ.\- ]')
birthplace_regex = re.compile('[^a-zA-Z0-9áÁéÉíÍóÓöÖőŐüÜúÚ\- ]')
date_regex = re.compile('[^0-9]')


def __run_tesseract_multiple_images(images, extension_configs, lang, run_otsu = False, blur_image = False, cluster_image = False):
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
        elif cluster_image:
            if blur_image:
                image = cv2.GaussianBlur(image, (5,5), 0.7)

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
            image = cv2.copyMakeBorder(image, 25, 25, 15, 15, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        elif blur_image:
            image = cv2.GaussianBlur(image, (5,5), 0.7)

        images[i] = image


    read_str = pytesseract.run_multiple_and_get_output(images, extension='tsv', extension_configs=extension_configs, config="--psm 7", lang=lang)
    #print("Tesseract time: " + str(time.time() - start_time))
    #for idx, image in enumerate(images):
        #cv2.imshow("TEST", image)
        #cv2.waitKey(0)
        #cv2.imwrite("test%d.png" % idx, image, (cv2.IMWRITE_PNG_COMPRESSION, 0))

    read_values = []
    lines = read_str.split("\n")
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


def __image_digits(read_values):
    date = ""
    confidence_level = 0
    text_index = read_values[0].index("text")
    confidence_index = read_values[0].index("conf")
    for i in range(1, len(read_values)):
        if len(read_values[i]) > text_index:
            name_part_candidate = date_regex.sub('', read_values[i][text_index]).lstrip().rstrip()
            if len(name_part_candidate) == 8:
                date = name_part_candidate
                confidence_level = int(read_values[i][confidence_index])

    if len(date) == 0:
        return ConfidenceValue(value="", confidence=0)

    year = int(date[0:4])
    if year < 1900:
        year = 1900 + int(date[2:4])
        confidence_level -= 10
    elif year > 2100:
        year = 2000 + int(date[2:4])
        confidence_level -= 10

    month = int(date[4:6])
    if 12 < month < 20:
        month = 10
        confidence_level -= 10
    elif month > 21:
        month = int(date[5:6])
        confidence_level -= 10

    day = int(date[6:])
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
    city = name.upper().rstrip().lstrip()

    city_parts = city.split(' ')
    for part in city_parts.copy()[::-1]:
        if len(part) == 0 or part[0] == '-':
            confidence_levels.pop(city_parts.index(part))
            city_parts.remove(part)

    best_candidate = ''
    numbers_found = ''

    for part in city_parts:
        if part in cities:
            best_candidate = part
            break
        elif len(part) >= 2 and part[0:2].isdigit() and 0 < int(part[0:2]) <= 23:
            numbers_found = part[0:2]
        elif len(part) > len(best_candidate):
            best_candidate = part

    confidence_level = confidence_levels[city_parts.index(best_candidate)]
    if best_candidate.title() == "Budapest":
        index = city_parts.index(best_candidate)
        if len(city_parts) > index + 1:
            numbers = city_parts[index + 1]
            if 0 < len(numbers) <= 2 and numbers.isdigit():
                best_candidate = best_candidate + ' ' + numbers
    elif best_candidate not in cities and len(numbers_found) == 2:
        best_candidate = 'Budapest ' + numbers_found

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
        if len(part) == 0 or '-' in part:
            city_parts.remove(part)

    best_candidate = ''
    for part in city_parts:
        if part in cities:
            return part.title()
        elif len(part) > len(best_candidate):
            best_candidate = part

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
        [[285, 450], [770, 550]], #barcode
        [[140, 140], [720, 220]], #name
        [[385, 280], [710, 325]], #birthplace
        [[210, 320], [745, 370]], #mother_name_primary
        [[65, 355], [455, 400]], #mother_name_secondary
        [[565, 385], [770, 435]], #release data
    ],
    [ #NEW_CARD
        [[335, 583], [616, 675]], #barcode
        [[100, 600], [200, 700]], #datamatrix
        [[255, 210], [705, 270]], #name first#
        [[255, 260], [705, 310]], #name_second
        [[360, 343], [605, 400]], #birthplace
        [[45, 436], [545, 490]], #mother name
        [[263, 480], [469, 540]] #release date
    ]
]


def __get_image_part(img, coordinates):
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


def __get_barcode_data(img):
    for i in range(1, 5):
        barcode_x_scale = 1 / i
        barcode_image = cv2.resize(img, (0, 0), fx=barcode_x_scale, fy=1)

        info = pyzbar.pyzbar.decode(barcode_image)
        if len(info) == 0:
            continue

        for barcode in info:
            if (barcode.type == "CODE128" or barcode.type == "I25") and __is_valid_id_num(barcode.data.decode("UTF-8")):
                return barcode.data.decode("UTF-8")

    return None


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


def __run_image_field_check(img, card_type, runlevel, unchecked_fields, validating_fields, found_fields, run_otsu, use_blur, use_cluster):

    name = birthplace = mother_name_1 = mother_name_2 = release_date = serial_number = None

    if runlevel == 0:
        if card_type == CardType.OLD_CARD:
            image_parts = []
            if "name" in unchecked_fields:
                image_parts.append(__get_image_part(img, field_coordinates[card_type][1]))
            if "birthplace" in unchecked_fields:
                image_parts.append(__get_image_part(img, field_coordinates[card_type][2]))
            if "mother_name" in unchecked_fields:
                image_parts.append(__get_image_part(img, field_coordinates[card_type][3]))
                image_parts.append(__get_image_part(img, field_coordinates[card_type][4]))
            if "release_date" in unchecked_fields:
                image_parts.append(__get_image_part(img, field_coordinates[card_type][5]))

            if len(image_parts) != 0:
                tesseract_output = __run_tesseract_multiple_images(image_parts, extension_configs=["bazaar_complete"],
                                                                   lang="hun_fast", run_otsu=run_otsu, blur_image=use_blur, cluster_image=use_cluster)
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
                mother_name_1, found_mother_1 = __image_name(tesseract_output[i])
                i += 1
                mother_name_2, found_mother_2 = __image_name(tesseract_output[i])
                i += 1
            if "release_date" in unchecked_fields:
                release_date = __image_digits(tesseract_output[i])
            serial_number = None
        else:
            image_parts = []
            if "name" in unchecked_fields:
                image_parts.append(__get_image_part(img, field_coordinates[card_type][2]))
                image_parts.append(__get_image_part(img, field_coordinates[card_type][3]))
            if "birthplace" in unchecked_fields:
                image_parts.append(__get_image_part(img, field_coordinates[card_type][4]))
            if "mother_name" in unchecked_fields:
                image_parts.append(__get_image_part(img, field_coordinates[card_type][5]))
            if "release_date" in unchecked_fields:
                image_parts.append(__get_image_part(img, field_coordinates[card_type][6]))

            if len(image_parts) != 0:
                tesseract_output = __run_tesseract_multiple_images(image_parts, extension_configs=["bazaar_complete"],
                                                                   lang="hun_fast", run_otsu=run_otsu, blur_image=use_blur, cluster_image=use_cluster)
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
                serial_number = __get_datamatrix_data(__get_image_part(img, field_coordinates[card_type][1]))

    else:
        if card_type == CardType.OLD_CARD:
            image_parts = []
            if "name" in unchecked_fields:
                image_parts.append(__get_image_part(img, field_coordinates[card_type][1]))
            if "mother_name" in unchecked_fields:
                image_parts.append(__get_image_part(img, field_coordinates[card_type][3]))
                image_parts.append(__get_image_part(img, field_coordinates[card_type][4]))

            if len(image_parts) != 0:
                tesseract_output = __run_tesseract_multiple_images(image_parts, extension_configs=["bazaar_name"],
                                                                   lang="hun_fast", run_otsu=run_otsu, blur_image=use_blur, cluster_image=use_cluster)
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
                    [__get_image_part(img, field_coordinates[card_type][2])
                     ], extension_configs=["bazaar_city"], lang="hun_fast", run_otsu=run_otsu)
                birthplace = __image_city(tesseract_output[0])

            if "release_date" in unchecked_fields:
                tesseract_output = __run_tesseract_multiple_images(
                    [__get_image_part(img, field_coordinates[card_type][5])
                     ], extension_configs=["digits"], lang="hun_fast", run_otsu=run_otsu)
                release_date = __image_digits(tesseract_output[0])

            serial_number = None
        else:
            image_parts = []
            if "name" in unchecked_fields:
                image_parts.append(__get_image_part(img, field_coordinates[card_type][2]))
                image_parts.append(__get_image_part(img, field_coordinates[card_type][3]))
            if "mother_name" in unchecked_fields:
                image_parts.append(__get_image_part(img, field_coordinates[card_type][5]))

            if len(image_parts) != 0:
                tesseract_output = __run_tesseract_multiple_images(image_parts, extension_configs=["bazaar_name"],
                                                                   lang="hun_fast", run_otsu=run_otsu, blur_image=use_blur, cluster_image=use_cluster)
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
                    [__get_image_part(img, field_coordinates[card_type][4])
                     ], extension_configs=["bazaar_city"], lang="hun_fast", run_otsu=run_otsu)
                birthplace = __image_city(tesseract_output[0])

            if "release_date" in unchecked_fields:
                tesseract_output = __run_tesseract_multiple_images(
                    [__get_image_part(img, field_coordinates[card_type][6])
                     ], extension_configs=["digits"], lang="hun_fast", run_otsu=run_otsu)
                release_date = __image_digits(tesseract_output[0])

            if "serial_number" in unchecked_fields:
                serial_number = __get_datamatrix_data(__get_image_part(img, field_coordinates[card_type][1]))

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


def validate_id_card(img, runlevel, validating_fields):
    validating_fields = dict(validating_fields)
    unchecked_fields = set()

    for key in validating_fields:
        unchecked_fields.add(key)

    ###read_id_card
    start_time = time.time()

    original_img = img

    if runlevel == 0:
        transform_target_width = 640
    else:
        transform_target_width = 1280

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

    #print("Warp time: " + str((time.time() - start_time)))

    # Barcode detection
    barcode_id_num = __get_barcode_data(__get_image_part(img, field_coordinates[card_type][0]))
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
                    barcode_id_num = __get_barcode_data(__get_image_part(img, field_coordinates[card_type][0]))
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
                        # cv2.imshow("Test", img)
                        # cv2.waitKey(0)
                        barcode_id_num = __get_barcode_data(__get_image_part(img, field_coordinates[card_type][0]))
                        if barcode_id_num is None:
                            return __get_barcode_response(original_img)
                    return __get_barcode_response(original_img)
                else:
                    barcode_id_num = __get_barcode_data(__get_image_part(img, field_coordinates[card_type][0]))
                    if barcode_id_num is None:
                        img = __get_transform_sift_for_type(original_img, CardType.NEW_CARD, transform_target_width,
                                                            runlevel)
                        card_type = CardType.NEW_CARD
                        if type(img) is str:
                            return __get_barcode_response(original_img)
                        else:
                            # cv2.imshow("Test", img)
                            # cv2.waitKey(0)
                            barcode_id_num = __get_barcode_data(__get_image_part(img, field_coordinates[card_type][0]))
                            if barcode_id_num is None:
                                return __get_barcode_response(original_img)
        else:
            return __get_barcode_response(original_img)

    real_id_num = barcode_id_num
    #print("Barcode time: " + str((time.time() - start_time)))

    real_birthday = ConfidenceValue(value=__birth_date_from_id(real_id_num).strftime('%Y.%m.%d'),
                                    confidence=random.randint(90, 100))

    if 'id_number' in unchecked_fields:
        unchecked_fields.remove('id_number')

    if 'birthdate' in unchecked_fields:
        unchecked_fields.remove('birthdate')
        if validating_fields['birthdate'][-1] == '.':
            validating_fields['birthdate'] = validating_fields['birthdate'][0:-1]

    found_fields = {}

    __run_image_field_check(img, card_type, runlevel, unchecked_fields, validating_fields, found_fields, run_otsu=False, use_blur=False, use_cluster=False)
    if len(unchecked_fields) != 0:
        __run_image_field_check(img, card_type, runlevel, unchecked_fields, validating_fields, found_fields, run_otsu=True, use_blur=False, use_cluster=False)
    if len(unchecked_fields) != 0:
        __run_image_field_check(img, card_type, runlevel, unchecked_fields, validating_fields, found_fields, run_otsu=False, use_blur=False, use_cluster=True)
    if len(unchecked_fields) != 0:
        __run_image_field_check(img, card_type, runlevel, unchecked_fields, validating_fields, found_fields, run_otsu=False, use_blur=True, use_cluster=True)
    if len(unchecked_fields) != 0:
        __run_image_field_check(img, card_type, runlevel, unchecked_fields, validating_fields, found_fields, run_otsu=True, use_blur=True, use_cluster=False)
    if len(unchecked_fields) != 0:
        __run_image_field_check(img, card_type, runlevel, unchecked_fields, validating_fields, found_fields, run_otsu=False, use_blur=True, use_cluster=False)

    if card_type == CardType.OLD_CARD and len(unchecked_fields) != 0 and alternate_card_used == False:
        img = __get_transform_sift_for_type(original_img, CardType.OLD_CARD, transform_target_width, runlevel, use_alternate_card=True)
        __run_image_field_check(img, card_type, runlevel, unchecked_fields, validating_fields, found_fields,
                                run_otsu=False, use_blur=False, use_cluster=False)
        if len(unchecked_fields) != 0:
            __run_image_field_check(img, card_type, runlevel, unchecked_fields, validating_fields, found_fields,
                                    run_otsu=True, use_blur=False, use_cluster=False)
        if len(unchecked_fields) != 0:
            __run_image_field_check(img, card_type, runlevel, unchecked_fields, validating_fields, found_fields,
                                    run_otsu=False, use_blur=False, use_cluster=True)
        if len(unchecked_fields) != 0:
            __run_image_field_check(img, card_type, runlevel, unchecked_fields, validating_fields, found_fields,
                                    run_otsu=False, use_blur=True, use_cluster=True)
        if len(unchecked_fields) != 0:
            __run_image_field_check(img, card_type, runlevel, unchecked_fields, validating_fields, found_fields,
                                    run_otsu=True, use_blur=True, use_cluster=False)
        if len(unchecked_fields) != 0:
            __run_image_field_check(img, card_type, runlevel, unchecked_fields, validating_fields, found_fields,
                                    run_otsu=False, use_blur=True, use_cluster=False)


    valid_failed = False
    valid_success = True

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
        elif found_fields[key] is not None:
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
    return response


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
    barcode_id_num = __get_barcode_data(__get_image_part(img, field_coordinates[card_type][0]))
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
                    barcode_id_num = __get_barcode_data(__get_image_part(img, field_coordinates[card_type][0]))
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
                        barcode_id_num = __get_barcode_data(__get_image_part(img, field_coordinates[card_type][0]))
                        if barcode_id_num is None:
                            return __get_barcode_response(original_img)
                    return __get_barcode_response(original_img)
                else:
                    barcode_id_num = __get_barcode_data(__get_image_part(img, field_coordinates[card_type][0]))
                    if barcode_id_num is None:
                        img = __get_transform_sift_for_type(original_img, CardType.NEW_CARD, transform_target_width,
                                                            runlevel)
                        card_type = CardType.NEW_CARD
                        if type(img) is str:
                            return __get_barcode_response(original_img)
                        else:
                            # cv2.imshow("Test", img)
                            # cv2.waitKey(0)
                            barcode_id_num = __get_barcode_data(__get_image_part(img, field_coordinates[card_type][0]))
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
    #    cv2.imwrite("test_output/test" + str(i) + ".jpg", __get_image_part(img, field_coordinates[card_type][i]))

    name = birthplace = mother_name_1 = mother_name_2 = release_date = serial_number = None
    found_name = found_mother_1 = found_mother_2 = False

    if runlevel == 0 or runlevel == 1:
        if card_type == CardType.OLD_CARD:
            image_parts = []
            if "name" not in filter_fields:
                image_parts.append(__get_image_part(img, field_coordinates[card_type][1]))
            if "birthplace" not in filter_fields:
                image_parts.append(__get_image_part(img, field_coordinates[card_type][2]))
            if "mother_name" not in filter_fields:
                image_parts.append(__get_image_part(img, field_coordinates[card_type][3]))
                image_parts.append(__get_image_part(img, field_coordinates[card_type][4]))
            if "release_date" not in filter_fields:
                image_parts.append(__get_image_part(img, field_coordinates[card_type][5]))

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
                image_parts.append(__get_image_part(img, field_coordinates[card_type][2]))
                image_parts.append(__get_image_part(img, field_coordinates[card_type][3]))
            if "birthplace" not in filter_fields:
                image_parts.append(__get_image_part(img, field_coordinates[card_type][4]))
            if "mother_name" not in filter_fields:
                image_parts.append(__get_image_part(img, field_coordinates[card_type][5]))
            if "release_date" not in filter_fields:
                image_parts.append(__get_image_part(img, field_coordinates[card_type][6]))

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
                serial_number = __get_datamatrix_data(__get_image_part(img, field_coordinates[card_type][1]))

    else:
        if card_type == CardType.OLD_CARD:
            image_parts = []
            if "name" not in filter_fields:
                image_parts.append(__get_image_part(img, field_coordinates[card_type][1]))
            if "mother_name" not in filter_fields:
                image_parts.append(__get_image_part(img, field_coordinates[card_type][3]))
                image_parts.append(__get_image_part(img, field_coordinates[card_type][4]))

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
                tesseract_output = __run_tesseract_multiple_images([__get_image_part(img, field_coordinates[card_type][2])
                                                                    ], extension_configs=["bazaar_city"], lang="hun_fast", run_otsu=run_otsu, blur_image=use_blur)
                birthplace = __image_city(tesseract_output[0])

            if "release_date" not in filter_fields:
                tesseract_output = __run_tesseract_multiple_images([__get_image_part(img, field_coordinates[card_type][5])
                                                                    ], extension_configs=["digits"], lang="hun_fast", run_otsu=run_otsu, blur_image=use_blur)
                release_date = __image_digits(tesseract_output[0])

            serial_number = None
        else:
            image_parts = []
            if "name" not in filter_fields:
                image_parts.append(__get_image_part(img, field_coordinates[card_type][2]))
                image_parts.append(__get_image_part(img, field_coordinates[card_type][3]))
            if "mother_name" not in filter_fields:
                image_parts.append(__get_image_part(img, field_coordinates[card_type][5]))

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
                tesseract_output = __run_tesseract_multiple_images([__get_image_part(img, field_coordinates[card_type][4])
                                                                    ], extension_configs=["bazaar_city"], lang="hun_fast", run_otsu=run_otsu, blur_image=use_blur)
                birthplace = __image_city(tesseract_output[0])

            if "release_date" not in filter_fields:
                tesseract_output = __run_tesseract_multiple_images([__get_image_part(img, field_coordinates[card_type][6])
                                                                    ], extension_configs=["digits"], lang="hun_fast", run_otsu=run_otsu, blur_image=use_blur)
                release_date = __image_digits(tesseract_output[0])

            if "serial_number" not in filter_fields:
                serial_number = __get_datamatrix_data(__get_image_part(img, field_coordinates[card_type][1]))


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