"""
python test.py --input xxx.jpg
"""

from PIL import Image
from PIL import ImageDraw
import argparse

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', required=True,
                      help='File path of image to process.')
    args = parser.parse_args()

    image = Image.open(args.input)
    image.show()

if __name__ == '__main__':
  main()


