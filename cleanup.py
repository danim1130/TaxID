import os
import time

def main():
    while True:
        path = r'/tmp/'
        now = time.time()
        for f in os.listdir(path):
            if f[0:5] == "tess_" and os.stat(os.path.join(path, f)).st_mtime < now - 5 * 60:
                os.remove(os.path.join(path, f))
        time.sleep(10 * 60)

if __name__ == "__main__":
    main()
