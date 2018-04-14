""" Dedicated module for standardising the size and shape information of a series of images.
Images with the same dimensions are essential for the Modeller class to function.
"""
# Author: Alex Jones <alexjonesphd@gmail.com>

import cv2
import math

import skimage.io
import numpy as np

def _convert_LMs(tem_file):
    """User accessible version of Modeller's get_LMs function.
    """

    with open(tem_file, 'r') as file:

        # Convert the first line of the template file into an integer
        n_points = int( next( file ) )

        # Loop through remainder of the file, strip newlines, split by space for each line
        LMs = [ line.strip().split() for line in file ]

    # Slice this list up to the number of points, and set as np.ndarray
    lm_array = np.array( LMs[ 0:n_points ], dtype = 'float' )

    # Capture line connection information
    try:
        line_info = LMs[n_points:]
    except:
        line_info = None

    return lm_array, line_info

def _tem_writer(tem_name, shape, line_data):
    """ User accessible version of Modellers tem_writer function, for saving templates.
    """

    with open(tem_name, 'w') as tem_file:

        # Write the numner of points by calculating the length, or number of points, of the predicted values
        tem_file.write('%s\n' % len (shape))

        # Loop over the array and write out - the predicted matrix is now a list-of-lists
        for line in shape.astype('str').tolist():

            # Join lines with tab and add new line
            tem_file.write('\t'.join( line ) + '\n')

        # Write the line data string provided in line_info using similar method as above, calling the object attribute here
        for line_value in line_data:

            # Make string to be printed, then print
            tem_file.write('%s\n' % ' '.join(line_value) )


def _similarity_transform(in_points, out_points):
    """ Given a set of two points, compute a similarity transform that will allow the full array of points
    to be roated and scaled to fit the new output. cv2 requires a third point that will be computed/generated from the input points.
    """

    # Convert input arrays to lists
    in_points = in_points.tolist()
    out_points = out_points.tolist()


    # Compute angles for third point - assume an equilateral triangle
    s60 = math.sin(60*math.pi/180)
    c60 = math.cos(60*math.pi/180)

    # Make up the third point for the input points
    xin = c60*(in_points[0][0] - in_points[1][0]) - s60*(in_points[0][1] - in_points[1][1]) + in_points[1][0]
    yin = s60*(in_points[0][0] - in_points[1][0]) + c60*(in_points[0][1] - in_points[1][1]) + in_points[1][1]

    in_points.append([np.int(xin), np.int(yin)])

    # Make up the third point for the input points
    xout = c60*(out_points[0][0] - out_points[1][0]) - s60*(out_points[0][1] - out_points[1][1]) + out_points[1][0]
    yout = s60*(out_points[0][0] - out_points[1][0]) + c60*(out_points[0][1] - out_points[1][1]) + out_points[1][1]

    out_points.append([np.int(xout), np.int(yout)])

    # Estime the transform needed
    tform = cv2.estimateRigidTransform(np.array([in_points]), np.array([out_points]), False)

    return tform


def standardise(master_data_frame, pt1_index = 0, pt2_index = 1, width = 500, height = 700, overwrite = False):
    """ Carries out standardisation of images listed in the index of a dataframe, without extensions.
    Carries out a similarity transform, to position the pupils in rouhgly the same location withn in an image.
    The index of the eye points from the template are specified as 0 and as defaults, as standard within Psychomorph.
    Original files can be overwritten with the overwrite argument.
    Desired output dimensions can also be specified using height and width inputs, and depending on the size and variation of the images as input,
    variations may need to be carried out to get all the images within the frame.
    """

    # Compute a rough  point for the centre of the eyes in each image
    estimate = [(0.35 * width, 0.4 * height), (0.65 * width, 0.4 * height)]
    estimate = np.array(estimate)

    # Iterate over the data frame and extract the image and template information
    for face in master_data_frame.index:

        # Read in shape information
        shape, line_info = _convert_LMs(face + '.tem')

        # Read in image
        im = skimage.io.imread(face + '.jpg')

        # Get this face's points
        eye_points = shape[[pt1_index, pt2_index], :]

        # Compute the similarity transform for this face's eye points and the estimate
        tform = _similarity_transform(eye_points, estimate)

        # Apply transform to the image data
        aligned_im = cv2.warpAffine(im, tform, (width, height))

        # Apply transform to the shape information
        aligned_shape = cv2.transform(shape.reshape(len(shape), 1, 2), tform)

        # Restructure shape coords
        aligned_shape = aligned_shape.reshape(len(aligned_shape), 2)

        # Save the output
        if overwrite:
            skimage.io.imsave(face + '.jpg', aligned_im)
            _tem_writer(face + '.tem', aligned_shape, line_info)
        else:
            skimage.io.imsave(face + '_aligned.jpg', aligned_im)
            _tem_writer(face + '_aligned.tem', aligned_shape, line_info)
