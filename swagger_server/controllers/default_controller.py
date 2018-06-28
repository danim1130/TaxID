import subprocess
import tempfile

import cv2
import numpy as np
import os

import id_scripts.id_card as id_card
from swagger_server.models import ConfidenceValue, CheckResponse
from swagger_server.models.error import Error  # noqa: E501
from swagger_server.models.tax_id_card import TaxIdCard  # noqa: E501


with open("api_keys.txt") as f:
    content = f.readlines()

content = frozenset([x.strip() for x in content])


def __gs_pdf_to_png(pdf):
    temp_name = tempfile.mktemp(prefix='tess_pdfjpeg_')
    pdf_file = temp_name + ".pdf"
    try:
        with open(pdf_file, "wb") as file:
            file.write(pdf)

        arglist = ["gs",
                   "-dBATCH",
                   "-dNOPAUSE",
                   "-sOutputFile=%s.jpeg" % temp_name,
                   "-sDEVICE=jpeg",
                   "-dFirstPage=1",
                   "-dLastPage=1",
                   "-dJPEGQ=100",
                   "-r100",
                   pdf_file]
        subprocess.run(arglist)

        image = cv2.imread("%s.jpeg" % temp_name)
        return image
    except:
        print("Error during Ghostscript")
    finally:
        try:
            os.remove(pdf_file)
            os.remove("%s.jpeg" % temp_name)
        except:
            print("Exception during cleanup")
    return None

def __extract_jpg_from_pdf(pdf):
    startmark = b"\xff\xd8"
    startfix = 0
    endmark = b"\xff\xd9"
    endfix = 2
    i = 0

    while True:
        istream = pdf.find(b"stream", i)
        if istream < 0:
            break
        istart = pdf.find(startmark, istream, istream + 20)
        if istart < 0:
            i = istream + 20
            continue
        iend = pdf.find(b"endstream", istart)
        if iend < 0:
            return None
        iend = pdf.find(endmark, iend - 20)
        if iend < 0:
            return None

        istart += startfix
        iend += endfix
        jpg = pdf[istart:iend]
        return jpg

    return None


def __get_image_from_stream(image, runlevel):
    if image.filename.split(".")[-1] == "pdf":
        pdf = image.stream.read()
        img = __extract_jpg_from_pdf(pdf)
        if img is None:
            img = __gs_pdf_to_png(pdf)
        else:
            img = cv2.imdecode(np.fromstring(img, np.uint8), cv2.IMREAD_ANYCOLOR)
    else:
        nparr = np.fromstring(image.stream.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)

    return img

def check_id_post(image, apiKey, name=None, birthdate=None, mothername=None, releasedate=None, id_num=None, birthplace=None, runlevel=None):  # noqa: E501
    """Checks submitted information

    The endpoint checks the submitted information against the submitted image. # noqa: E501

    :param image:
    :type image: werkzeug.datastructures.FileStorage
    :param name:
    :type name: str
    :param birthdate:
    :type birthdate: str
    :param mothername:
    :type mothername: str
    :param releasedate:
    :type releasedate: str
    :param id_num:
    :type id_num: str
    :param birthplace:
    :type birthplace: str
    :param runlevel: Lower value means faster runtime, but lower precision.
    :type runlevel: int

    :rtype: CheckResponse
    """
    if apiKey not in content:
        return Error("Invalid ApiKey provided!")

    if runlevel is None:
        runlevel = 0

    img = __get_image_from_stream(image, runlevel)
    if type(img) is Error:
        return img

    validate_fields = {}
    if name is not None:
        validate_fields['name'] = name.strip()
    if birthdate is not None:
        validate_fields['birthdate'] = birthdate
    if mothername is not None:
        validate_fields['mother_name'] = mothername
    if releasedate is not None:
        validate_fields['release_date'] = releasedate
    if id_num is not None:
        validate_fields['id_number'] = id_num
    if birthplace is not None:
        validate_fields['birthplace'] = birthplace

    card = id_card.validate_id_card(img, runlevel, validate_fields)
    if type(card) is str:
        return Error(card)
    else:

        if card['validation_failed']:
            result = 'reject'
        elif card['validation_success']:
            result = 'accept'
        else:
            result = 'validate'

        return CheckResponse(
            id_num=None if id_num is None else card['id_number'],
            birthdate=None if birthdate is None else card['birthdate'],
            name=None if name is None else card['name'],
            mother_name=None if mothername is None else card['mother_name'],
            release_date=None if releasedate is None else card['release_date'],
            birthplace=None if birthplace is None else card['birthplace'],
            validation_result=result
        )

def read_id_post(image, apiKey, runlevel=None):  # noqa: E501
    """Extract information from the submitted card

    The endpoint returns information about the submitted taxId card. Only supports PNG format. # noqa: E501

    :param image: 
    :type image: werkzeug.datastructures.FileStorage

    :rtype: TaxIdCard
    """

    if apiKey not in content:
        return Error("Invalid ApiKey provided!")

    if runlevel is None:
        runlevel = 0

    img = __get_image_from_stream(image, runlevel)
    if type(img) is Error:
        return img

    card = id_card.read_id_card(img, runlevel, run_otsu=False)
    card2 = id_card.read_id_card(img, runlevel, run_otsu=True)

    if type(card) is str:
        ret = Error(card)
    elif type(card2) is str:
        ret = Error(card2)
    else:
        ret = [TaxIdCard(id_num=card.id_num,
                        birthdate=ConfidenceValue(card.birthday.value, card.birthday.confidence),
                        birthplace=ConfidenceValue(card.birthplace.value, card.birthplace.confidence),
                        name=ConfidenceValue(card.name.value, card.name.confidence),
                        mother_name=ConfidenceValue(card.mother_name_primary.value, card.mother_name_primary.confidence),
                        release_date=ConfidenceValue(card.release_date.value, card.release_date.confidence),
                        valid=card.valid,
                        type=card.type,
                        serial_num=card.serial_number),
               TaxIdCard(id_num=card2.id_num,
                         birthdate=ConfidenceValue(card2.birthday.value, card2.birthday.confidence),
                         birthplace=ConfidenceValue(card2.birthplace.value, card2.birthplace.confidence),
                         name=ConfidenceValue(card2.name.value, card2.name.confidence),
                         mother_name=ConfidenceValue(card2.mother_name_primary.value,
                                                     card2.mother_name_primary.confidence),
                         release_date=ConfidenceValue(card2.release_date.value, card2.release_date.confidence),
                         valid=card2.valid,
                         type=card2.type,
                         serial_num=card2.serial_number)]

    return ret


def test_cpu_get():  # noqa: E501
    try:
        with open("/proc/cpuinfo", "r") as cpu:
            return cpu.read()
    except IOError as e:
        return Error("I/O error({0}): {1}".format(e.errno, e.strerror))
