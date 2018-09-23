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
    """ Object to model relationships between facial appearance and multiple predictors. Upon instantiation, the class requires a Pandas dataframe where the index column is a string describing a face that should match with a template AND image in the current folder,
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
        save_warp : During gathering, each image will be warped to the average facial shape. If a user wants to see these faces, setting this parameter
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


    def assemble_multivariate(self, parameter_key, gathered_data = None, as_frame = False):
        """Convenience function to extract from the gathered data and assemble the face parameters - shape and colour - and predictor values into a pair of arrays.
        Each array is matched on the rows to represent measurements of that face parameter and the predictors, for a subject.
        
        Parameters
        ----------
        parameter_key : A string used to specify the parameter of interest, which will access the gathered_data dictionary. Can take the following properties:
                        'shape' - to access shape landmarks
                        'channel_one' - access colour information in channel one
                        'channel_two' - access colour information in channel two
                        'channel_three' - access colour information in channel three
                        
        as_frame : Boolean determining whether the resulting arrays will be a NumPy array or a Pandas DataFrame. Default is False.

        Returns
        ----------
        y_array : An array containing the measured parameters for the faces, either shape or colour values, a row per observation. Data is extracted from the gathered data.
        x_array : An array containing predictor values, matched on the rows to the faces. 
        """

        if gathered_data is None:
            gathered_data = self.gathered_data
        
        assert parameter_key in ['shape', 'channel_one', 'channel_two', 'channel_three'], "Invalid parameter key; must be 'shape', 'channel_one', 'channel_two', 'channel_three'."
        if parameter_key == 'shape':
            parameter_dims = self.template_dims
        else:
            parameter_dims = self.image_dims

        # Compute the dimensions of arrays #

        # N obs is the number of rows in master data
        n_obs = self.master_data_frame.shape[0]

        # x_dims is the number of IV's, so columns of master data
        x_dims = self.master_data_frame.shape[1]

        # y_dims will be the product of the dimensions of shape or colour data
        y_dims = np.prod(parameter_dims)

        # Preallocate the arrays
        y_array = np.empty((n_obs, y_dims))
        x_array = np.empty((n_obs, x_dims))
        face_id = []

        # Iterate over the dictionary values and extract, using enumerate to index preallocated arrays
        # The order the faces are returned is not important, but locking the paramater to the IV's is vital
        for index, (face, values) in enumerate(gathered_data.items()):

            # Extract Y values, flatten and store
            y_array[index, :] = values[parameter_key].flatten()

            # Extract X values in exact order by iterating over trait list, which stores traits in order they appear in master
            x_array[index, :] = [values[trait] for trait in self.trait_list]
            
            # Store face id
            face_id.append(face)

        # Set up output
        if as_frame:
            y_array = pd.DataFrame(data=y_array, index=face_id)
            x_array = pd.DataFrame(data=x_array, index=face_id, columns=self.trait_list)
                                   
        return y_array, x_array


    def __calculate_slopes_intercept(self, y_array, x_array):
        """ Calculates the slopes and intercepts for the predictors and features given. To speed things up drastically, coefficients are calculated
        using NumPy linear algebra functions, which are extremely well optimised.
        """

        # Add in a column of ones to the input predictors to provide an estimate of the intercept.
        constant_term = np.column_stack((np.ones(x_array.shape[0]), x_array))

        # Using matrix algebra, decompose into linear function - first row is intercept of all values, second row onwards are slopes for each point/pixel
        coefs, residual, _, _ = np.linalg.lstsq(constant_term, y_array, rcond = -1)

        # Calculate residual degrees of freedom of regression - N samples minus constant term's number of coefs (predictors plus intercept term)
        df_total = x_array.shape[0] - 1
        df_model = x_array.shape[1] - 1 # Subtract one for intercept
        df_error = df_total - df_model

        # Get the slope and intercept by indexing
        intercept = coefs[0,:] # First row is the intercept
        slopes = coefs[1:,:] # From second row to end are slopes

        # Finally compute the standard error of the predictions, sqrt of residual divided by residual DF
        stand_err = np.sqrt(residual/df_error)

        return slopes, intercept, stand_err


    def fit(self, template_dims = None, image_dims = None, n_preds = None):
        """ The workhorse of the class. Fits a regression to each element of the shape and texture data, producing a multidimensional array that
        represents the slopes and intercepts of each element (e.g. red pixel (0, 0), y coordinate of landmark number 4, etc). This can later be utilised
        by the `.predict` function produce predicted faces.

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

        # Compute weights and intercepts #

        # Prepare least squares arrays
        shape_y, shape_x = self.assemble_multivariate('shape')
        channel_one_y, channel_one_x = self.assemble_multivariate('channel_one')
        channel_two_y, channel_two_x = self.assemble_multivariate('channel_two')
        channel_three_y, channel_three_x = self.assemble_multivariate('channel_three')

        # Compute least squares solutions
        shape_slopes, shape_intercepts, shape_se = self.__calculate_slopes_intercept(shape_y, shape_x)
        red_channel_slopes, red_channel_intercepts, red_channel_se = self.__calculate_slopes_intercept(channel_one_y, channel_one_x)
        green_channel_slopes, green_channel_intercepts, green_channel_se = self.__calculate_slopes_intercept(channel_two_y, channel_two_x)
        blue_channel_slopes, blue_channel_intercepts, blue_channel_se = self.__calculate_slopes_intercept(channel_three_y, channel_three_x)

        # Reshape to the dimensions of the model's inputs
        self.shape_slopes = np.swapaxes(shape_slopes,0,1).reshape(n_landmarks, n_dims, n_preds)
        self.shape_intercepts = shape_intercepts.reshape(n_landmarks, n_dims)
        self.shape_se = shape_se.reshape(n_landmarks, n_dims)

        self.red_channel_slopes = np.swapaxes(red_channel_slopes, 0, 1).reshape(im_height, im_width, n_preds)
        self.red_channel_intercepts = red_channel_intercepts.reshape(im_height, im_width)
        self.red_channel_se = red_channel_se.reshape(im_height, im_width)

        self.green_channel_slopes = np.swapaxes(green_channel_slopes, 0, 1).reshape(im_height, im_width, n_preds)
        self.green_channel_intercepts = green_channel_intercepts.reshape(im_height, im_width)
        self.green_channel_se = green_channel_se.reshape(im_height, im_width)

        self.blue_channel_slopes = np.swapaxes(blue_channel_slopes, 0, 1).reshape(im_height, im_width, n_preds)
        self.blue_channel_intercepts = blue_channel_intercepts.reshape(im_height, im_width)
        self.blue_channel_se = blue_channel_se.reshape(im_height, im_width)

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
