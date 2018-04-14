""" Saving and loading functions for storing and retrieving objects created by the Modeller class.
"""
# Author: Alex Jones <alexjonesphd@gmail.com>

import pickle


def model_save(modeller_object, filename = None):
    """Saves the object to a pickle file, so object does not need to be refitted after calling
    gather_data and fit methods of the Modeller class once.

    Parameters
    ----------
    modeller_object : An instance of the Modeller class, ideally with gathered_data and fit called.
    filename :  A filename for the saved object. .pkl extension is auto-appended by this function.
    """
    # Auto append the pickle extension to the filename
    filename += '.pkl'

    # Save
    with open(filename, 'wb') as output:
        pickle.dump(modeller_object, output)


def model_load(pickled_model = None):
    """Simple function to load saved models that have been pickled into the workspace.

    Parameters
    ----------
    pickled_model : A filename for the model to be loaded. .pkl extension is auto-appended.

    Returns
    ----------
    loaded : A deserialised model, complete with gathered data and fitted arrays, if they were called before saving.
    """
    # Auto append the pickle extension to the filename
    pickled_model += '.pkl'

    # Load
    with open(pickled_model, 'rb') as input_model:
        loaded = pickle.load(input_model)

    return loaded
