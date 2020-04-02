# Local Feature Stencil Code
# Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech with Henry Hu <henryhu@gatech.edu>
# Edited by James Tompkin
# Adapted for python by asabel and jdemari1 (2019)

import argparse

import matplotlib
import numpy as np

from skimage import io, img_as_float32
from skimage.transform import rescale
from skimage.color import rgb2gray

import student as student
from helpers import evaluate_correspondence, cheat_interest_points

matplotlib.use("TkAgg")


# This script
# (1) Loads and resizes images
# (2) Finds interest points in those images                 (you code this)
# (3) Describes each interest point with a local feature    (you code this)
# (4) Finds matching features                               (you code this)
# (5) Visualizes the matches
# (6) Evaluates the matches based on ground truth correspondences

def load_data(file_name):
    """
     1) Load stuff
     There are numerous other image sets in the supplementary data on the
     project web page. You can simply download images off the Internet, as
     well. However, the evaluation function at the bottom of this script will
     only work for three particular image pairs (unless you add ground truth
     annotations for other image pairs). It is suggested that you only work
     with the two Notre Dame images until you are satisfied with your
     implementation and ready to test on additional images. A single scale
     pipeline works fine for these two images (and will give you full credit
     for this project), but you will need local features at multiple scales to
     handle harder cases.

     If you want to add new images to test, create a new elif of the same format as those
     for notre_dame, mt_rushmore, etc. You do not need to set the eval_file variable unless
     you hand create a ground truth annotations. To run with your new images use
     python main.py -p <your file name>.

    :param file_name: string for which image pair to compute correspondence for

        The first three strings can be used as shortcuts to the
        data files we give you

        1. notre_dame
        2. mt_rushmore
        3. e_gaudi

    :return: a tuple of the format (image1, image2, eval file)
    """

    # Note: these files default to notre dame, unless otherwise specified
    image1_file = "../data/NotreDame/NotreDame1.jpg"
    image2_file = "../data/NotreDame/NotreDame2.jpg"

    eval_file = "../data/NotreDame/NotreDameEval.mat"

    if file_name == "notre_dame":
        pass
    elif file_name == "mt_rushmore":
        image1_file = "../data/MountRushmore/Mount_Rushmore1.jpg"
        image2_file = "../data/MountRushmore/Mount_Rushmore2.jpg"
        eval_file = "../data/MountRushmore/MountRushmoreEval.mat"
    elif file_name == "e_gaudi":
        image1_file = "../data/EpiscopalGaudi/EGaudi_1.jpg"
        image2_file = "../data/EpiscopalGaudi/EGaudi_2.jpg"
        eval_file = "../data/EpiscopalGaudi/EGaudiEval.mat"

    image1 = img_as_float32(io.imread(image1_file))
    image2 = img_as_float32(io.imread(image2_file))

    return image1, image2, eval_file


def main():
    """
    Reads in the data,

    Command line usage: python main.py -p | --pair <image pair name>

    -p | --pair - flag - required. specifies which image pair to match

    """

    # create the command line parser
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--pair", required=True,
                        help="Either notre_dame, mt_rushmore, or e_gaudi. Specifies which image pair to match")

    args = parser.parse_args()

    # (1) Load in the data
    image1_color, image2_color, eval_file = load_data(args.pair)

    # You don't have to work with grayscale images. Matching with color
    # information might be helpful. If you choose to work with RGB images, just
    # comment these two lines

    image1 = rgb2gray(image1_color)
    # Our own rgb2gray coefficients which match Rec.ITU-R BT.601-7 (NTSC) luminance conversion - only mino
    # performance improvements and could be confusing to students image1 = image1[:,:,0] * 0.2989 + image1[:,:,
    # 1] * 0.5870 + image1[:,:,2] * 0.1140
    image2 = rgb2gray(image2_color)
    # image2 = image2[:,:,0] * 0.2989 + image2[:,:,1] * 0.5870 + image2[:,:,2] * 0.1140

    # make images smaller to speed up the algorithm. This parameter
    # gets passed into the evaluation code, so don't resize the images
    # except for changing this parameter - We will evaluate your code using
    # scale_factor = 0.5, so be aware of this
    scale_factor = 0.5

    # Bilinear rescaling
    image1 = np.float32(rescale(image1, scale_factor))
    image2 = np.float32(rescale(image2, scale_factor))

    # width and height of each local feature, in pixels
    feature_width = 16

    # (2) Find distinctive points in each image. See Szeliski 4.1.1
    # !!! You will need to implement get_interest_points. !!!

    print("Getting interest points...")

    (x1, y1) = student.get_interest_points(image1, feature_width)
    (x2, y2) = student.get_interest_points(image2, feature_width)

    # For development and debugging get_features and match_features, you will likely
    # want to use the ta ground truth points, you can comment out the preceding two
    # lines and uncomment the following line to do this. Note that the ground truth
    # points for mt. rushmore will not produce good results, so you'll have to use
    # your own function for that image pair.

    # (x1, y1, x2, y2) = cheat_interest_points(eval_file, scale_factor)

    # if you want to view your corners uncomment these next lines!

    # plt.imshow(image1, cmap="gray")
    # plt.scatter(x1, y1, alpha=0.9, s=3)
    # plt.show()

    # plt.imshow(image2, cmap="gray")
    # plt.scatter(x2, y2, alpha=0.9, s=3)
    # plt.show()

    print("Done!")

    # 3) Create feature vectors at each interest point. Szeliski 4.1.2
    # !!! You will need to implement get_features. !!!

    print("Getting features...")

    image1_features = student.get_features(image1, x1, y1, feature_width)
    image2_features = student.get_features(image2, x2, y2, feature_width)

    print("Done!")

    # 4) Match features. Szeliski 4.1.3
    # !!! You will need to implement match_features !!!

    print("Matching features...")

    matches, confidences = student.match_features(image1_features, image2_features)

    print("Done!")

    # 5) Evaluation and visualization

    # The last thing to do is to check how your code performs on the image pairs
    # we've provided. The evaluate_correspondence function below will print out
    # the accuracy of your feature matching for your 50 most confident matches,
    # 100 most confident matches, and all your matches. It will then visualize
    # the matches by drawing green lines between points for correct matches and
    # red lines for incorrect matches. The visualizer will show the top
    # num_pts_to_visualize most confident matches, so feel free to change the
    # parameter to whatever you like.

    print("Matches: " + str(matches.shape[0]))

    num_pts_to_visualize = 50

    evaluate_correspondence(image1_color, image2_color, eval_file, scale_factor,
                            x1, y1, x2, y2, matches, confidences, num_pts_to_visualize, args.pair + '_matches.jpg')


if __name__ == '__main__':
    main()
