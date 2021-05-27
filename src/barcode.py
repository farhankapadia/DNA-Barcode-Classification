# importing image object from PIL
import math
from PIL import Image, ImageDraw

def barcodeimage(newSeq):
    fillStyle = 0
    newSeq = newSeq.upper()

    # # Setup the colors per nucleotides
    kAdenineColor = "rgb(0,248,0)";		# A is Green
    kGuanineColor = "rgb(0,0,0)";		# G is Black
    kCytosineColor = "rgb(0,0,248)";	# C is Blue
    kThymineColor = "rgb(255,0,0)";		# T is Red

    w, h = 1, 40

    j =0;
    length = len(newSeq) - newSeq.count("N")
    
    # creating new Image object
    img = Image.new("RGB", (w*length, h))


    for i in range(0,len(newSeq)):

        # Every base is drawn as 1 barcode bar
        if (newSeq[i] == 'A'):
            fillStyle = kAdenineColor

        elif (newSeq[i] == 'C'):
            fillStyle = kCytosineColor

        elif (newSeq[i] == 'G'):
            fillStyle = kGuanineColor

        elif (newSeq[i] == 'T' or newSeq[i] == 'U'):
            fillStyle = kThymineColor
        else:
            fillStyle = 'none'


        if( fillStyle != 'none'):

            # create rectangle image
            ImageDraw.Draw(img).rectangle(((0 + (w*j), 0), (w+ (w*j), h)), fill = fillStyle)
            j = j + 1


    # img.show()
    img.save("barcode.png")
