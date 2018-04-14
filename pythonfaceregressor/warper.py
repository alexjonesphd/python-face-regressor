""" Dedicated module for carrying out image warping between two images, and a set of shape coordinates.
Computes a Delaunay Triangulation for the set of shape coordinates, then warps each triangle from image 1 to image 2 by a set amount, alpha.
Used primarily to move individual faces to and from the average within the Modeller class, but can be used separately by calling the warp function and providing images and coordinates.
Requires OpenCv for Delaunay algorithm and triangle deformations.
"""
# Author: Alex Jones <alexjonesphd@gmail.com>

import cv2
import numpy as np

def _tem_to_tuple(template):
    """ Convert a .tem file into a list of tuples for use with OpenCV functions.
    Not used within the Modeller class, but useful for standalone use.
    """

    with open(template, 'rb') as tem_file:

        # Collect the number of points, specified at the start of the template
        n_points = int(next(tem_file))

        # Read in the rest of the file into a variable, but splitting and stripping characters
        file_info = [line.strip().split() for line in tem_file]

        # Set up lambda to apply integer and float conversion in one
        convert = lambda x: int(float(x))

        # Iterate over list up to npoints, and map the lambda function to each coordinate point
        XY = [ tuple(map(convert, point)) for point in file_info[:n_points] ]
        
        # Extract line information to append later
        line_info = file_info[n_points:]

    return XY, line_info

def _add_surround(image, points):
    """ For better warping results, a set of eight points are imputed across the image borders before triangulation,
    calculated within this function and added to the point list providedself.
    """

    # Get dimensions of image
    height, width, _ = image.shape

    # Subtract 1 from height and width to truncate below maximum values
    height_ = height - 1
    width_ = width - 1

    # Generate eight points  that match the top and bottom corners and middle of each rectangle side
    surround_points = [(0, 0), (width//2, 0), (width_, 0),
                       (0, height//2), (width_, height//2),
                       (0, height_), (width//2, height_), (width_, height_)]

    appended = points + surround_points

    return appended

def _generate_triangles(image, points):
    """ Applies the Delaunay Triangulation on the given set of points, and returns a list containing the index locations of each point
    that make up the triangles. Yielding the index, rather than the coordinates directly, is desirable as the algorithm will find different triangles
    on the same set of points even if they are very similar.
    """

    # Get size of image and generate a rectangle the size of the image
    height, width, _ = image.shape
    im_rect = (0, 0, width, height)

    # Instantiate the subdivider object, set to the size of the image rectangle
    subdivider = cv2.Subdiv2D(im_rect)

    # Add all points into the subdivider object
    for point in points:
        subdivider.insert(point)

    # The triangle list can now be accessed, but some points are outside of the face image.
    # These need to be trimmed out, which can be done using a quick custom function defined below, to check whether a point is within the bounds of given image
    def in_image(rect, point):
        if point[0] < rect[0]:
            return False
        elif point[1] < rect[1]:
            return False
        elif point[0] > rect[2]:
            return False
        elif point[1] > rect[3]:
            return False

        return True

    # Define triangle index list to be returned
    delaunay_list = []

    # Iterate over the triangle list, set up the points and filter them using the in_image function
    for triangle in subdivider.getTriangleList():

        # Parse out points from list
        pt1 = (triangle[0], triangle[1])
        pt2 = (triangle[2], triangle[3])
        pt3 = (triangle[4], triangle[5])

        # Check if the point is within the image - if so, find its index in points and append
        if in_image(im_rect, pt1) and in_image(im_rect, pt2) and in_image(im_rect, pt3):

            # Find the index by passing each pt to the .index method of input point list
            ind1 = points.index(pt1)
            ind2 = points.index(pt2)
            ind3 = points.index(pt3)

            # Append the point indices
            delaunay_list.append([ind1, ind2, ind3])

    return delaunay_list


def _applyAffineTransform(src, srcTri, dstTri, size):
    """Applies the affine transformation from a source triangle to the destination triangle, effectively warping a segment of an image.
    """

    # Given a pair of triangles, find the affine transform
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst


def _morph_triangle(image1, image2, output_im, tri1, tri2, output_tri, alpha):
    """ Extracts small rectangles from each image in the same location that match a provided triangle, and warps them.
    Once warping is complete, fills in warped rectangle to the new image.
    """

    # Compute the bounding rectangle of each input triangle
    rect1 = cv2.boundingRect(np.float32(tri1))
    rect2 = cv2.boundingRect(np.float32(tri2))
    out_rect = cv2.boundingRect(np.float32(output_tri))

    # Each point needs to be offset by the top left corner of the bounding rectangle
    t1_rect = []
    t2_rect = []
    t_out_rect = []

    for tri1_pt, tri2_pt, output_pt in zip(tri1, tri2, output_tri):

        t1_rect.append( ( (tri1_pt[0] - rect1[0]), (tri1_pt[1] - rect1[1]) ) )
        t2_rect.append( ( (tri2_pt[0] - rect2[0]), (tri2_pt[1] - rect2[1]) ) )
        t_out_rect.append( ( (output_pt[0] - out_rect[0]), (output_pt[1] - out_rect[1]) ) )

    # Mask the output triangle to get it ready for warping, and then fill it
    mask = np.zeros((out_rect[3], out_rect[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_out_rect), (1.0, 1.0, 1.0), 16, 0 );

    # Crop out small rectangular patches from the input images using the calculated offset rectangles
    img1_crop = image1[ rect1[1]:rect1[1] + rect1[3], rect1[0]:rect1[0] + rect1[2] ]
    img2_crop = image2[ rect2[1]:rect2[1] + rect2[3], rect2[0]:rect2[0] + rect2[2] ]

    size = (out_rect[2], out_rect[3])
    warp_image1 = _applyAffineTransform(img1_crop, t1_rect, t_out_rect, size)
    warp_image2 = _applyAffineTransform(img2_crop, t2_rect, t_out_rect, size)

    # Perform the alpha blending on the resulting image patches
    img_rect = (1 - alpha) * warp_image1 + alpha * warp_image2

    # Copy the triangular region of the rectangle patch to the final image
    # For texture blend, use this line
    #output_im[ out_rect[1]:out_rect[1] + out_rect[3], out_rect[0]:out_rect[0] + out_rect[2] ] = output_im[ out_rect[1]:out_rect[1] + out_rect[3], out_rect[0]:out_rect[0] + out_rect[2] ] * ( 1 - mask ) + img_rect * mask

    # For shape only, simply add in the warped triangle from image 1.
    output_im[ out_rect[1]:out_rect[1] + out_rect[3], out_rect[0]:out_rect[0] + out_rect[2] ] = output_im[ out_rect[1]:out_rect[1] + out_rect[3], out_rect[0]:out_rect[0] + out_rect[2] ] * (1 - mask) + warp_image1 * mask

    return output_im


def warp(image1, points1, image2, points2, alpha = 1):
    """ General function that utilises functionality of the module. Takes in two images and their respective shape coordinates
    as well as an alpha argument (default = 1), or the amount to warp by. Returns a uint8-dtyped image to save or visualise.
    Image points should be in the form a numpy.ndarray of any data type.
    """

    # Convert input images to float32 for operations
    img1 = np.float32(image1)
    img2 = np.float32(image2)

    # Prepare a blank canvas for the output image
    out_img = np.zeros(img1.shape, dtype = img1.dtype)
    
    # Check the type of the input shape coordinates, and ensure they are the same length
    if isinstance(points1, np.ndarray) and isinstance(points2, np.ndarray):
        shape1 = [tuple(pt) for pt in points1.astype(int).tolist()]
        shape2 = [tuple(pt) for pt in points2.astype(int).tolist()]
        assert len(shape1) is len(shape2), 'Unequal number of landmarks for images'
    else:
        raise TypeError('Input coorindates are not NumPy arrays. Convert to np.ndarray after reading using _tem_to_tuple')

    # Compute the alpha-weighted shape that the output image will have
    out_points = []
    for shape1_pt, shape2_pt in zip(shape1, shape2):

        x = int( (1 - alpha) * shape1_pt[0] + alpha * shape2_pt[0] )
        y = int( (1 - alpha) * shape1_pt[1] + alpha * shape2_pt[1] )

        out_points.append((x,y))

    # Append the surround points for better warping
    shape1_app = _add_surround(img1, shape1)
    shape2_app = _add_surround(img2, shape2)
    out_points_app = _add_surround(out_img, out_points)

    # Obtain the indexed-Delaunay triangles for the weighted average points, so the other shapes can be accurately indexed, and not have the alogirthm recompute across each set of points
    delaunay_list = _generate_triangles(out_img, out_points_app)

    # With triangle list generated, iterate over and extract the points from each shape list, and pass to the _morph_triangle function for warping
    for (tr1, tr2, tr3) in delaunay_list:

        shape1_tri = [ shape1_app[tr1], shape1_app[tr2], shape1_app[tr3] ]
        shape2_tri = [ shape2_app[tr1], shape2_app[tr2], shape2_app[tr3] ]
        dest_tri = [ out_points_app[tr1], out_points_app[tr2], out_points_app[tr3] ]

        # Overwrite and pass back the out_img array
        out_img = _morph_triangle(img1, img2, out_img, shape1_tri, shape2_tri, dest_tri, alpha)

    # Return the warped image out.
    # NOTE - opencv works in BGR, not RGB, so returned image is actually BGR. However, matplotlib.imsave and imshow functions implicitly convert, so no need to explicitly change ordered here.
    # If needed - out_img[:,:,::-1]
    return np.uint8(out_img)
