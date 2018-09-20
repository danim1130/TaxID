FROM ubuntu:18.04

RUN apt-get update && apt-get install -y tesseract-ocr python3.6 python3-pip libsm6 libxext6 libzbar-dev poppler-utils libdmtx0a ghostscript libleptonica-dev libtesseract-dev && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN chmod -R 777 /tmp/

COPY tesseract_files/hun_fast.traineddata /usr/share/tesseract-ocr/4.00/tessdata/
COPY tesseract_files/hun_fast.user-words /usr/share/tesseract-ocr/4.00/tessdata/
COPY tesseract_files/bazaar_complete /usr/share/tesseract-ocr/4.00/tessdata/configs/

COPY tesseract_files/hun_precise.traineddata /usr/share/tesseract-ocr/4.00/tessdata/
COPY tesseract_files/hun_precise.user-words-name /usr/share/tesseract-ocr/4.00/tessdata/
COPY tesseract_files/hun_precise.user-words-city /usr/share/tesseract-ocr/4.00/tessdata/
COPY tesseract_files/bazaar_name /usr/share/tesseract-ocr/4.00/tessdata/configs/
COPY tesseract_files/bazaar_city /usr/share/tesseract-ocr/4.00/tessdata/configs/

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

RUN mkdir saved

COPY requirements.txt /usr/src/app/

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /usr/src/app

ENV OMP_THREAD_LIMIT=1

EXPOSE 8080

CMD python3.6 cleanup.py & gunicorn -b 0.0.0.0:8080 -w $(cat /proc/cpuinfo | grep processor | wc -l) --max-requests 4 --max-requests-jitter 4 --timeout 30 main

