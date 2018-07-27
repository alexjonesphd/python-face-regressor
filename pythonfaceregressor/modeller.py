"""
Modeller
----------
The workhorse of the package. The Modeller class provides a clean interface to carrying out multiple regression across a set of facial landmarks and pixels
with predictors associated with those same faces. Allows users to predict the appearance of faces by passing predictor weights in a multiple regression style formula.
"""
# Author: Alex Jones <alexjonesphd@gmail.com>

import numpy as np
import pandas as pd
import skimage.io

from scipy.stats import zscore

from . import warper

class Modeller():
    """ Object to model relationships between facial appearance and multiple predictors. Upon instantiation, the class requires a pandas dataframe where the index column is a string describing a face that should match with a template AND image in the current folder,
    Each column should be the values of the predictor(s), with a suitable header. The class methods then gather the data and creates an array of slopes that matches the size of the shape or image inputs,
    but with entries along axis 2 that map to the number of predictors.

    Parameters
    ----------
    master_data_frame : a pandas DataFrame object where the index column is a string matching the filenames of images and template files in a current directory, without extensions.
                        Each column should represent a predictor variable of interest, with a suitable header.

    standardise_impute : in the input DataFrame, some rows may have missing data or not be standardised. Given that linear regression works optimally with centred predictors and
                         no missing data, setting this argument to True will fill missing values with the mean of that predictor and apply a Z score transform to each variable.
                         Default is False.

    template_extension : a string to denote the extension for files containing shape coordinates. Default is '.tem', to work with Psychomorph.

    image_extension : a string to to denote the extension for image files. Default is '.jpg'.

    """

    def __init__(self, master_data_frame, standardise_impute = False, template_extension = '.tem', image_extension = '.jpg'):

        # Set file extensions to defaults, or what user specifies
        self.__template_extension = template_extension
        self.__image_extension = image_extension

        # Assert that the master_data_frame is a pandas DataFrame
        assert isinstance( master_data_frame, pd.DataFrame ), 'Input variable is not a pandas dataframe. Ensure input is dataframe, and index corresponds to image/template identifier, and columns are traits to be examined.'

        # Standardise and impute missing values if required
        if standardise_impute:
            master_data_frame = master_data_frame.fillna( master_data_frame.mean(axis = 0) ).transform( zscore, axis = 0, ddof = 1 )

        # Store the DataFrame as an attribute
        self.master_data_frame = master_data_frame

        # Prepare index-name mapping. This is a list of trait identifiers, and the numbers of the columns they are in. Stores the list itself and the mapping, which is useful later.
        traits = master_data_frame.columns.tolist()
        
        # Initialise a dict to store mappings of traits to index locations
        mapping = dict(zip(traits, range(0, len(traits))))

        # Set as attributes
        self.trait_list = traits
        self.mapping = mapping


    def __get_LMs(self, tem_file):
        """Convenience function to convert a template file into a numpy array.
        Returns the XY coordinates as an array, but also the trailing line data as a list of lists.
        """
        # Read in the template file with base Python
        with open( tem_file, 'r' ) as file:

            # Convert the first line of the template file into an integer
            n_points = int( next( file ) )

            # Loop through remainder of the file, strip newlines, split by space for each line
            LMs = [ line.strip().split() for line in file ]

        # Slice this list up to the number of points, and set as np.ndarray
        lm_array = np.array( LMs[ 0:n_points ], dtype = 'float' )

        # Slice the list to obtain the line data that follows the points - will return as a list-of-lists, else None
        try:
            line_info = LMs[n_points:]
        except:
            line_info = None

        return lm_array, line_info


    def gather_data(self, master_data_frame = None, template_extension = None, image_extension = None, save_warp = False):
        """
        Method to build the data necessary for the face regression.

        The folder the interpreter is pointing at should contain images and templates. This function gathers these files into a single dictionary,
        where the dictionary keys are the face identifier strings. Each key will return another dictionary that contains the score for that face on each predictor variable,
        as well as an array of coordinates and a separate array of each colour channel (RGB). Each face will be warped to the average face shape before being stored.

        Parameters
        ----------
        save_warp : During gathering, ecah image will be warped to the average facial shape. If a user wants to see these faces, setting this parameter
                    to True will save the warped version to disk.

        Returns
        ----------
        self.gathered_data : The modeller instance will not return objects, but will update self to store a dictionary of key-value pairs
                             where keys are identifier strings from the index of the input DataFrame, and values are another dictionary of predictors.
                             The predictor dictionary will have at least the following entries:

                             shape :  an m*n ndarray of (x,y) coordinates describing the shape of the face
                             channel_one : an m*n ndarray of pixel values describing the red/first channel of the image
                             channel_two : an m*n ndarray of pixel values describing the green/second channel of the image
                             channel_three : an m*n ndarray of pixel values describing the blue/third channel of the image

                             The predictor dictionary will also contain an entry, with the same name as the column header in the input DataFrame,
                             of the score on each predictor for a given face.
        """

        # Set the values to existing object attributes
        if master_data_frame is None:
            master_data_frame = self.master_data_frame

        if template_extension is None:
            template_extension = self.__template_extension
        if image_extension is None:
            image_extension = self.__image_extension

        # Preallocate the dictionary
        gathered = {}

        # Iterate over the index and extract the shape parameter of each face, which can be used to compute the average face shape for registration
        for face in master_data_frame.index:

            # Prepare the parameters that are entered into the dictionary #
            # Extract the landmarks from associated template file
            try:
                xy, line_data = self.__get_LMs( face + template_extension )
            except FileNotFoundError:
                raise ValueError( 'Missing template file for face: ' + face )

            gathered[face] = {'shape' : xy}

        # Compute the average face shape efficiently
        average_shape = np.dstack([gathered[face]['shape'] for face in gathered.keys()]).mean(axis = 2)

        # Iterate once more over the dataframe, using the keys to index images, and warp each face to the average shape, before segmenting and storing in dict along with trait data.
        for face in master_data_frame.index:

            # Read in colour image
            try:
                im = skimage.io.imread( face + image_extension )
            except:
                raise ValueError( 'Missing image file for face: ' + face )

            # Apply the image warping to standardise shape
            try:
                warp_im = warper.warp(im, gathered[face]['shape'], im, average_shape)
            except:
                print('Facial landmarks out of bounds for image {}. Continuing to gather data, but this image is not standardised. Recrop and align points, then re-gather to include.'.format( face ) )
                warp_im = im

            # Save if required
            if save_warp:
                skimage.io.imsave(face + '_aligned' + image_extension, warp_im)

            # Create an entry, and add in the parameters from the shape and image data above
            entry = {'channel_one' : warp_im[:,:,0], 'channel_two' : warp_im[:,:,1], 'channel_three' : warp_im[:,:,2]  }

            # Extract the trait information from the master data frame as a dictionary
            traits = master_data_frame.loc[ face ].to_dict()

            # Update the entry to hold the trait information
            entry.update( traits )

            # Finally store in the dictionary alongside the earlier shape information
            gathered[ face ].update(entry)

        # Check all arrays are the same size. Cheat a little by calling set on the returned shape arrays from the dictionary - if this is greater than 1, we know we have a different shape somewhere
        assert len( set( gathered[ face ][ 'shape' ].shape for face in gathered ) ) == 1, 'Input shape arrays are not all identical in dimensions. Check template files before calling .fit().'
        assert len( set( gathered[ face ][ 'channel_one' ].shape for face in gathered ) ) == 1, 'Input image arrays are not all identical in dimensions. Check image files before calling .fit().'

        # Set attributes - store sizes of shape and image arrays
        self.template_dims = gathered[ face ][ 'shape' ].shape
        self.image_dims = gathered[ face ][ 'channel_one' ].shape

        # Give the object access to the line data
        self.line_data = line_data

        # Give the object access to average face shape
        self.average_shape = average_shape
        
        # Compute the average face appearance and give object access
        R = np.dstack( [gathered[face]['channel_one'] for face in gathered] ).mean(axis = 2)
        G = np.dstack( [gathered[face]['channel_two'] for face in gathered] ).mean(axis = 2)
        B = np.dstack( [gathered[face]['channel_three'] for face in gathered] ).mean(axis = 2)
        image = np.dstack( (R, G, B) )
        
        self.average_face = image.astype('uint8')

        # Set the gathered dictionary data as an attribute
        self.gathered_data = gathered

        return self


    def __dictionary_extractor(self, index_value, parameter_key, predictors, gathered_data = None):
        """Extracts predictors from the dictionary, creating a matrix of predictors and a vector to regress against.
        How this function is called will depend on the inputs that are provided to the FIT function, described below. It will need to return shape or
        Red, Green, Blue values from the dictiionary along with traits."""

        if gathered_data is None:
            gathered_data = self.gathered_data

        # Preallocate the list to hold the retrieved values from the dictionary
        retrieved_values = []

        # Iterate over the dictionary keys (face IDs) and retrieve the required parameter data and trait values
        for faces in gathered_data:

            # Create container list that will grow with the requested values from the user: parameter is the first entry
            this_face_data = []

            # From the face key, pull the parameter (shape or colour channel), then pull the index value, e.g. point 0,0 or point 0,1 etc
            this_face_data.append( gathered_data[faces][parameter_key][index_value] )

            # With shape coordinate or pixel value, retrieve the traits specified in *traits, iterating over them and appending to this_face_data
            for trait in predictors:
                this_face_data.append( gathered_data[faces][trait]  )

            # Finally, append this_face_data to our master container, retrieved_values.
            retrieved_values.append(this_face_data)

        # Convert to numpy array for indexing and return
        point_data = np.array( retrieved_values )

        # Segment for ease of understanding, the first column is the parameter
        y = point_data[:,0]
        features = point_data[:,1:] # And the rest are the features

        return y, features


    def __calculate_slopes_intercept(self, y, features):
        """ Calculates the slopes and intercepts for the predictors and features given. To speed things up drastically, coefficients are calculated
        using NumPy linear algebra functions, which are extremely well optimised and shave off a lot of time. Returns a  1-D vector of slopes, which is length as the number of features, and a single intercept value.
        Also returns a standard error of predictions, which can be used for computing significance of predictions.
        """

        # Add in a column of ones to the input features to provide an estimate of the intercept.
        constant_term = np.column_stack( ( features, np.ones( len(features) ) ) )

        # Now decompose into a list of slopes, same length as the list of features, where the final element is the intercept, and compute the residual of predictions
        coefs, residual, _, _ = np.linalg.lstsq(constant_term, y, rcond = None)

        # Calculate residual degrees of freedom of regression - N samples minus constant term's number of coefs (predictors plus intercept term)
        deg_free = len(y) - len(coefs)

        # Get the slope and intercept by indexing
        slopes = coefs[:-1] # From start to end minus one are slopes
        intercept = coefs[-1] # The final index is the intercept

        # Finally compute the standard error of the predictions, sqrt of residual divided by residual DF
        stand_err = np.sqrt(residual/deg_free)

        return slopes, intercept, stand_err


    def fit(self, template_dims = None, image_dims = None, n_preds = None):
        """ The workhorse of the class. Fits a regression to each element of the shape and texture data, producing a multidimensional array that
        represents the slopes and intercepts of each element (e.g. red pixel (0, 0), y coordinate of landmark number 4, etc). This can later be modified
        via the predict function produce predicted shapes and textures.
        WARNING: This call will be slow for most images above 200 * 200 pixels. Please be patient and save once fitted.

        Returns
        ----------
        Once fit has successfully run, the modeller instance will update to give user access to a range of arrays describing various parameters of the model.

        self.shape_slopes : A l*m*n array, where axis 0 is the number of landmarks, axis 1 is the number of dimensions (2 or 3) and axis 2 is the number of predictors.
                            Entries in this array represent the regression weight of that point for that predictor.
        self.shape_intercepts : A m*n array, where axis 0 is the number of landmarks, axis 1 is the number of dimensions (2 or 3).
                            Entries in this array represent the regression intercept for that point. If standardise_impute was called, this value would be the average value in this location.
        self.shape_se : A m*n array, where axis 0 is the number of landmarks, axis 1 is the number of dimensions (2 or 3).
                            Entries in this array represent the standard error of the estimate for the regression equation - the mean accuracy of the prediction in that location, in units of the predicted values.

        The above attributes are also available for the three colour channels, describing image properties. For example:

        self.red_channel_slopes: A l*m*n array, where axis 0 is the number of landmarks, axis 1 is the number of dimensions (2 or 3) and axis 2 is the number of predictors.
                            Entries in this array represent the regression weight of that pixel value, for that predictor.

        """

        if template_dims is None:
            template_dims = self.template_dims
            n_landmarks, n_dims = template_dims

        if image_dims is None:
            image_dims = self.image_dims
            im_height, im_width = image_dims

        if n_preds is None:
            n_preds = len(self.trait_list) # Get the number of traits to predict

        # Preallocate all the arrays we need to fill through the fit function - one for slopes, one for intercept - for shapes AND textures
        shape_slopes = np.empty( ( n_landmarks, n_dims, n_preds ) )
        shape_intercepts = np.empty( ( n_landmarks, n_dims ) )
        shape_se = np.empty( ( n_landmarks, n_dims ) )

        red_channel_slopes = np.empty( ( im_height, im_width, n_preds )  )
        red_channel_intercepts = np.empty( ( im_height, im_width ) )
        red_channel_se = np.empty( ( im_height, im_width ) )

        green_channel_slopes = np.empty( ( im_height, im_width, n_preds )  )
        green_channel_intercepts = np.empty( ( im_height, im_width ) )
        green_channel_se = np.empty( ( im_height, im_width ) )

        blue_channel_slopes = np.empty( ( im_height, im_width, n_preds )  )
        blue_channel_intercepts = np.empty( ( im_height, im_width ) )
        blue_channel_se = np.empty( ( im_height, im_width ) )

        # First, fit the shapes through iterating over the values and extracting the relevant point, fitting the model, and saving the slope, intercepts, and standard error
        for index, _ in np.ndenumerate(shape_intercepts): # iterating over an array same size as the inputs

            # Update user on progress
            print( 'Predicting landmark coorindate: ' + str(index) )

            # Unpack the index for readability
            r, c = index

            # Extract the feature and outcome measure for this landmark value. Pass the attributes trait list and gathered data from __init__ and gather_data
            y, features = self.__dictionary_extractor(index, 'shape', self.trait_list, self.gathered_data)

            # Calculate the list of slopes and scalar intercept
            slopes, intercepts, stander = self.__calculate_slopes_intercept(y, features)

            # Store the slope, intercept, and se in the same index location
            shape_slopes[r, c, :] = slopes
            shape_intercepts[r, c] = intercepts
            shape_se[r, c] = stander

        # Set these as attributes
        self.shape_slopes = shape_slopes
        self.shape_intercepts = shape_intercepts
        self.shape_se = shape_se

        # Repeat the above process for colour
        for index, _ in np.ndenumerate(red_channel_intercepts): # Iterate over 2d structure same size as input images
            # Update user of progress, since this can be a very slow procedure for large images
            print( 'Predicting pixel location ' + str(index) )

            # Unpack index for clarity
            r, c = index

            # Extract the pixel values and the traits for each of the three channels in one go
            red_y, red_features = self.__dictionary_extractor(index, 'channel_one', self.trait_list, self.gathered_data)
            green_y, green_features = self.__dictionary_extractor(index, 'channel_two', self.trait_list, self.gathered_data)
            blue_y, blue_features = self.__dictionary_extractor(index, 'channel_three', self.trait_list, self.gathered_data)

            # Calculate the slope and intercept values across the three arrays
            red_slopes, red_intercept, red_stander = self.__calculate_slopes_intercept(red_y, red_features)
            green_slopes, green_intercept, green_stander = self.__calculate_slopes_intercept(green_y, green_features)
            blue_slopes, blue_intercept, blue_stander = self.__calculate_slopes_intercept(blue_y, blue_features)

            # Store these in the correct position
            red_channel_intercepts[r, c] = red_intercept
            green_channel_intercepts[r, c] = green_intercept
            blue_channel_intercepts[r, c] = blue_intercept

            red_channel_slopes[r, c, :] = red_slopes
            green_channel_slopes[r, c, :] = green_slopes
            blue_channel_slopes[r, c, :] = blue_slopes

            red_channel_se[r, c] = red_stander
            green_channel_se[r, c] = green_stander
            blue_channel_se[r, c] = blue_stander

        # Once prediction is finished, set as attributes
        self.red_channel_intercepts = red_channel_intercepts
        self.green_channel_intercepts = green_channel_intercepts
        self.blue_channel_intercepts = blue_channel_intercepts

        self.red_channel_slopes = red_channel_slopes
        self.green_channel_slopes = green_channel_slopes
        self.blue_channel_slopes = blue_channel_slopes

        self.red_channel_se = red_channel_se
        self.green_channel_se = green_channel_se
        self.blue_channel_se = blue_channel_se

        return self


    def predict(self, average_shape = None, shape_slopes = None, shape_intercepts = None,
                red_slopes = None, red_intercepts = None,
                green_slopes = None, green_intercepts = None,
                blue_slopes = None, blue_intercepts = None,
                mapping = None, **predict_values):
        """ Method that produces predicted faces from the fitted data, for users to save or interact with.

        Parameters
        ----------
        **predict_values : Keyword and values that represent the predictors and the desired weight to predict by.
                           These must match the column headers used in the DataFrame the Modeller object was instantiated with,
                           as these are inherited. Keywords and values can be passed as a dictionary. 

                           Example assuming predictors are 'Sex' and 'Attractiveness':

                           Standard usage - Modeller.predict(Sex = 1, Attractiveness = 4)
                           Dictionary usage - Modeller.predict({'Sex':1, 'Attractiveness':4})
                           
                           Any parameters that are omitted will contribute to the final appearance, though by a small amount. 
                           To remove influences of a particular trait, set its value to zero explicitly.


        Returns
        ----------
        formula : A string that reports the keyword and values entered into the function.
        final_image : l*m*n ndarray representing the final predicted face, warped to its corresponding shape
        final_shape :  m*n ndarray representing the predicted x,y coordinates
        texture_image : l*m*n ndarray representing the predicted pixel values of a facial appearance, retaining the average facial shape
        """

        if mapping is None:
            mapping = self.mapping

        if average_shape is None:
            average_shape = self.average_shape


        # COPY the fitted data below, as broadcasting will modify the original arrays due to Python pass-by-reference
        if shape_slopes is None:
            shape_slopes = np.copy( self.shape_slopes )

        if red_slopes is None:
            red_slopes = np.copy( self.red_channel_slopes )
        if green_slopes is None:
            green_slopes = np.copy( self.green_channel_slopes )
        if blue_slopes is None:
            blue_slopes = np.copy( self.blue_channel_slopes )

        if shape_intercepts is None:
            shape_intercepts = np.copy( self.shape_intercepts )

        if red_intercepts is None:
            red_intercepts = np.copy( self.red_channel_intercepts )
        if green_intercepts is None:
            green_intercepts = np.copy( self.green_channel_intercepts )
        if blue_intercepts is None:
            blue_intercepts = np.copy( self.blue_channel_intercepts )


        # Create an empty list to return the formula string, or traits that were specified to be multiplied by a constant
        form_str = []

        # Carry out broadcasting operations over slope matrix to multiple weights by the constant we want. Uses two dictionaries with matching keys to extract the trait-slope location (e.g. which value along axis 2) and the amount to broadcast by.
        for traits in predict_values: # Iterate over the keywords provided

            # Retrieve the index from the index map - so, if user provided 'Extra' for example, this will extract 'Extra' from the index map and the dimension where Extraversion is stored in the slope matrix
            index = mapping[ traits ]

            # Retrieve the broadcast factor again using the keyword provided, this will extract the value to be predicted from the keyword dictionary.
            broadcast = predict_values[ traits ]

            # Make the string and append it, joining with an underscore
            form_str.append( traits + '_' + str( broadcast ) )

            # Carry out broadcasting - finally, this will take the index and slice the slope_matrix array, broadcasting the predicted value across all slopes and reassigning it.
            shape_slopes[:, :, index] *= broadcast
            red_slopes[:, :, index] *= broadcast
            green_slopes[:, :, index] *= broadcast
            blue_slopes[:, :, index] *= broadcast

        # Find axis to sum along - can therefore handle two and three dim arrays, as multiple or linear regression, depending on number of trait inputs
        ax = red_slopes.ndim - 1

        # Compute the sum across the slopes, and add on the intercept matrix, to complete the regression formula and obtain predicted values
        final_shape = shape_slopes.sum( axis = ax ) + shape_intercepts

        final_red = red_slopes.sum( axis = ax ) + red_intercepts
        final_green = green_slopes.sum( axis = ax ) + green_intercepts
        final_blue = blue_slopes.sum( axis = ax ) + blue_intercepts

        # Combine colour channels into one, set dtype
        texture_image = np.dstack( (final_red, final_green, final_blue) ).astype('uint8')

        # Warp the texure image to average shape
        final_image = warper.warp(texture_image, average_shape, texture_image, final_shape)

        # Produce a clean string to aid in saving data
        formula = '_'.join(form_str)

        return formula, final_image, final_shape, texture_image


    def save_predictions(self, formula_string, predicted_texture, predicted_shape, line_data = None):
        """ Method to save the predicted images and associated shape files. Method will automatically append the trailing line data from a .tem file.
        Inputs can be passed directly from the .predict method, and image and shape file extensions are automatically inferred from input data.

        Parameters
        ----------
        formula_string : A string describing the desired filename of the image. It is convenient to pass the formula string returned from .predict.
        predicted_texture : Image array to write out. This can be the warped image or the shape-invariant image, both produced by .predict.
        predicted_shape : Array of shape coordinates to be written to file. These are produced by .predict.
        line_data : Values describing lines that join x,y coordinates in a Psychomorph template file. Default is None, but is inferred from input .tem files automatically.
        """

        # Set the line information
        if line_data is None:
            line_data = self.line_data

        # Set the output strings using the objects extension properties
        tem_name = formula_string + self.__template_extension
        image_name = formula_string + self.__image_extension

        # Check the predicted inputs and tidy up for output - formatting the templates and clipping the images
        predicted_shape = predicted_shape.round(decimals = 3).astype('str').tolist()
        np.clip( predicted_texture, 0, 255, out = predicted_texture )

        # Write the image out
        skimage.io.imsave( image_name, predicted_texture )

        # Write the template file
        with open( tem_name, 'w' ) as tem_file:

            # Write the numner of points by calculating the length, or number of points, of the predicted values
            tem_file.write( '%s\n' % len ( predicted_shape ) )

            # Loop over the array and write out - the predicted matrix is now a list-of-lists
            for line in predicted_shape:

                # Join lines with tab and add new line
                tem_file.write( '\t'.join( line ) + '\n' )

            # Write the line data string provided in line_info using similar method as above, calling the object attribute here
            for line_value in line_data:

                # Make string to be printed, then print
                tem_file.write( ' '.join( line_value ) + '\n' )

        return None
