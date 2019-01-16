from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import round
from builtins import range
from builtins import int
from future import standard_library
standard_library.install_aliases()
import cv2
import numpy as np
import gripit.edgelib.util as util
import matplotlib as plt
import copy
from gripit.edgelib.crop_image import crop_image


if __name__ == '__main__':
    for num_img in [4]:
        # Crops the image according to the user's mouse clicks
        # First click is top left, second click is bottom right
        mouse_X, mouse_Y = crop_image(num_img)

        # Read in depth image, -1 means w/ alpha channel.
        # This keeps in the holes with no depth data as just black.

        depth_im = 'img/learn%d.png' % num_img
        old_img = cv2.imread(depth_im, -1)

        # crops the depth image
        img = old_img[mouse_Y[0]:mouse_Y[1], mouse_X[0]:mouse_X[1]]
        cy = round(img.shape[0] / 2)
        cx = round(img.shape[1] / 2)

        print("x", img.shape[1], "y", img.shape[0])
        print("cy", cy, "cx", cx)

        P = {
             "img": img,
             "height": img.shape[0],
             "width": img.shape[1],
             "img_size": img.shape,
             "mouse_X": mouse_X,
             "mouse_Y": mouse_Y,
             "cx": cx,
             "cy": cy,
             "focal_length": 300,
             }

        # Values that depend on values in P
        P2 = {"blank_image": np.zeros((P["height"], P["width"], 3), np.uint8)}

        # adds all these new values to P
        P.update(P2)

        # Creates point cloud map
        # point_cloud = util.depth_to_PC(P)
        def depth_to_PC(P):
            # img is the depth image, blank_image is for the pointcloud
            ###Need to edit for mouseY and mouseX(crop it later)
            new_blank_image = copy.deepcopy(P["blank_image"])

            x_val = []
            y_val = []
            z_val = []

            for y_coord in range(len(new_blank_image)):
                for x_coord in range(len(new_blank_image[0])):
                    x, y, z = depth_to_3d(x_coord, y_coord, P)
                    """z = img[yCoord][xCoord]
                                                        x = (xCoord - cx) * z / f
                                                        y = (yCoord - cy) * z / f"""
                    # print(y, x)
                    # print("blank_imageyx", blank_image[yCoord][xCoord])
                    new_blank_image[y_coord][x_coord] = (x, y, z)
                    """newX.append(int(x))
                                                        newY.append(int(y))
                                                        newZ.append(int(z))"""
                    if (x_coord % 10 == 0 and y_coord % 10 == 0):
                        x_val.append(int(x))
                        y_val.append(int(y))
                        z_val.append(int(z))

            # create3dPlot in plot_3d.py (xVal, yVal, zVal)
            np.save("saveX", x_val)
            np.save("saveY", y_val)
            np.save("saveZ", z_val)

            return new_blank_image


        # DepthTo3d
        def depth_to_3d(x, y, P):
            cx = P["cx"]
            cy = P["cy"]
            f = P["focal_length"]
            z = copy.deepcopy(P["img"][y][x])
            x = (x - cx) * (z / (f))
            y = (y - cy) * (z / (f))
            return x, y, z

        point_cloud = depth_to_PC(P)
        print(point_cloud.shape)