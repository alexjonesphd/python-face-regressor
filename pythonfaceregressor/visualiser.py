""" Dedicated module for providing interactive visualisation of fitted relationships generated in the Modeller class.
Leverages Bokeh for visualisation, and the main 'explore' function will only run in a Jupyter Notebook.
"""
# Author: Alex Jones <alexjonesphd@gmail.com>

import PIL.Image
import numpy as np

from bokeh.models import ColumnDataSource, HoverTool, Slider
from bokeh.layouts import widgetbox, gridplot, layout
from bokeh.plotting import figure
from bokeh.io import show, output_notebook

from . import warper

def _image_tidy(face):
    """Tidies images predicted from Modeller class to fit with Bokeh's HTML RGBA representation.
    Converts and reshapes image.
    """

    # Get size of input image
    height, width, _ = face.shape

    # Turn the image into an RGB PIL object, ensuring it is uint8 and RGB typed
    im = PIL.Image.fromarray(face.astype('uint8'), 'RGB')

    # Resize to width * height and turn into RGBA
    im = im.resize((width, height)).convert('RGBA')

    # Change back to numpy array representation
    im_array = np.array(im)

    # Stack each pixel tuple as an RGBA in uint32, reshape to height * width
    final_image = im_array.view(dtype = np.uint32).reshape((height, width))

    # Reverse the y axis manually, because Bokeh plots backwards like Matplotlib
    final_image = final_image[::-1]

    return final_image


def explore(model):
    """ Allows a user to explore the model outputs using Bokeh, using sliders to adjust predicted shape and texture variables.
    This function will NOT run outside of a Jupyter Notebook.
    """

    # Set up the notebook output
    output_notebook()

    def make_document(doc):
        """ Internal function to create the document for visualisation. Hidden from users who need only call outer function with a fitted model.
        """

        # Use the model to predict an average representationof  data, using the mean value of each predictor
        centred_predict = model.master_data_frame.mean().to_dict()
        _, final_im, shape, texture_im = model.predict(**centred_predict)

        # Call the image tidy function to change RGB texture to RGBA
        final_im = _image_tidy(final_im)
        texture = _image_tidy(texture_im)

        # Set up the plot for the shape coordinates ###############################
        # Define a column data source
        shape_source = ColumnDataSource( {'x': shape[:,0], 'y': shape[:,1]} )

        # Instantiate plot object for shape coordinates
        shape_plot = figure(title = 'Predicted Shape', y_range = (900, 0), x_range = (0, 900))
        shape_plot.cross('x', 'y', size = 10, source = shape_source)

        # Define hover tool and add to plot
        hover = HoverTool( tooltips = [('x', '@x'), ('y', '@y')] )
        shape_plot.add_tools(hover)
        ###########################################################################

        # Set up a column data source for the actual warped face ##################
        # Define a column data source
        warp_source = ColumnDataSource( {'image': [final_im]} )

        # Instantiate plot object for warped image - add a constant extra few pixels to make sure image is not squashed to window
        warp_image_plot = figure(title = 'Predicted Face', y_range = (0, model.image_dims[0]+300), x_range = (0, model.image_dims[1]+300))
        warp_image_plot.image_rgba(image = 'image', x=0, y=0, dw=model.image_dims[1], dh=model.image_dims[0], source=warp_source)

        # Set up a column data source for the texture-only face ###################
        # Define a column data source
        texture_source = ColumnDataSource( { 'image': [texture] } )

        # Instantiate plot object for shape-free face
        image_plot = figure(title = 'Predicted Texture', y_range = (0, model.image_dims[0]+300), x_range = (0, model.image_dims[1]+300) )
        image_plot.image_rgba( image = 'image', x=0, y=0, dw=model.image_dims[1], dh=model.image_dims[0], source=texture_source)
        ###########################################################################

        # Define the internal callback function to update objects interactively
        def callback(attr, old, new):
            """ Bokeh callback for updating glyphs
            """

            # Iterate over the traits, get their title and their value and store in the dictionary
            predictor_dict = {}
            for slide in sliders:
                predictor_dict[ slide.title ] = slide.value

            # Use this dictionary to feed to the model's predict method, generating new ouput to show
            _, final_im, shape, texture = model.predict(**predictor_dict)

            # Fix the images for show
            final_im = _image_tidy(final_im)
            texture = _image_tidy(texture)

            # Update data sources with the new information
            shape_source.data = {'x':shape[:,0], 'y':shape[:,1]}
            warp_source.data = {'image':[final_im]}
            texture_source.data = {'image':[texture]}

        ###########################################################################
        # Set up sliders to alter properties
        sliders = []
        for trait in model.trait_list:

            #  Get the middle and far end points by applying mean, min, and max, and rounding to zero
            avg, mini, maxi = model.master_data_frame[trait].apply(['mean', 'min', 'max']).round()

            slider = Slider(title = trait, start = mini, end = maxi, step = 1, value = avg)
            slider.on_change('value', callback)
            sliders.append(slider)

        ###########################################################################

        # Set layout
        layout = gridplot([widgetbox(sliders), warp_image_plot], [image_plot, shape_plot])

        # Update and add to curdoc
        doc.add_root(layout)

    # Initialise server with the make_document function defined above
    show(make_document)
