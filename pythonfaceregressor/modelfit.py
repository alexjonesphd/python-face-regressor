"""
ModelFit
----------
Conduct multivariate statistical analysis of a fitted Modeller instance. The ModelFit class will produce a permuted non parametric F test of each of the elements
of faces - shape, and one for each of the three colour channels.
# Author: Alex Jones <alexjonesphd@gmail.com>
"""

import numpy as np
import pandas as pd

from . import Modeller


class ModelFit():
    """Object to carry out statistical analysis of fitted datasets. Upon instantiation, the class requires an instance of a Modeller class that has had
    the `.gather_data` method already called on it.

    Parameters
    ----------
    fitted_instance :   an instance of Modeller that has had `.gather_data()` called on it, so the object is aware of the dataset to be tested for significance.
    fit_order : ModelFit currently only supports sequential tests of predictors. Use `fit_order` to specify a list of strings representing variable names in the order to be tested.
    """

    def __init__(self, fitted_instance, fit_order=None):

        # Assert the passed input is a PFR Modeller instance
        assert isinstance(fitted_instance, Modeller), 'Input needs to be a fitted face regressor Modeller class.'

        # Make sure that it has the `gathered_data` attribute
        assert hasattr(fitted_instance, 'gathered_data'), 'Call `gather_data` before attempting a model fit.'

        # Prepare the arrays
        shape, texture, predictors = fitted_instance.return_arrays(as_frame=True)

        # Reorder predictors according to focus list
        if fit_order:
            predictors = predictors[fit_order]

        # Parse texture into channels
        red = texture.loc[:, texture.columns.str.contains('Ch1')]
        green = texture.loc[:, texture.columns.str.contains('Ch2')]
        blue = texture.loc[:, texture.columns.str.contains('Ch3')]

        # Add a constant to the predictors and re-arrange it so it is the first column
        predictors.insert(0, 'const', np.ones(predictors.shape[0]))

        # Function to quickly mean - centre a large array as the images should be
        def mc(x):
            return x - x.mean()

        # Convert image arrays to full PC representation - trace is slow to compute otherwise
        def pca_projection(centred_arr):
            U, S, V = np.linalg.svd(centred_arr, full_matrices=False)
            comps = U * S
            return comps

        # Set attributes to centred data and store predictors
        self.shape_data = np.apply_along_axis(mc, 0, shape.values)
        self.red_data = pca_projection(np.apply_along_axis(mc, 0, red.values))
        self.green_data = pca_projection(np.apply_along_axis(mc, 0, green.values))
        self.blue_data = pca_projection(np.apply_along_axis(mc, 0, blue.values))
        self.preds = predictors

        return None

    def fit_all(self, n_preds=999):
        """Carries out the permuted analysis of variance on the data with the psuedo F test.
        Predictors will be added to the model in the order specified by fit_order on instantiation.

        Parameters
        ----------

        n_preds : an integer representing the number of permutations to conduct. More is more computationally expensive.


        Returns
        -------

        full : a dictionary containing four DataFrames containing the model fits, psuedo F and R square values for each of the aspects of face data.
               Keys are ['shape', 'Ch1', 'Ch2', 'Ch3'].
        """

        # Simply call and return all permuted fits
        shape_fit = self._permute_fit(self.preds, self.shape_data, n_preds=n_preds)
        red_fit = self._permute_fit(self.preds, self.red_data, n_preds=n_preds)
        green_fit = self._permute_fit(self.preds, self.green_data, n_preds=n_preds)
        blue_fit = self._permute_fit(self.preds, self.blue_data, n_preds=n_preds)

        # Return
        full = {'Shape': shape_fit, 'Ch1': red_fit, 'Ch2': green_fit, 'Ch3': blue_fit}

        return full

    def _permute_fit(self, predictors, y_array, n_preds=999):
        """Carries out the permutations of the data and refits analysis n_perms times.
        """

        # Obtain actual F ratios via multivariate fitting
        original_data = self._sequential_fit(predictors, y_array)

        # Carry out permutations
        storage = {key: [] for key in predictors.drop('const', 1).columns}

        for p in range(n_preds):

            # Shuffle
            shuffled = predictors.sample(frac=1)

            # Use this to compute a sequential fit
            permuted_seq_fit = self._sequential_fit(shuffled, y_array)

            # Subset the results and turn it into a dictionary
            perm_vals = dict(permuted_seq_fit['F Ratio'])

            # Extract and store against all other permutations
            [storage[key].append(value) for key, value in perm_vals.items()]

        # Compute P values
        pvals = {}
        for trait, f in dict(original_data['F Ratio']).items():

            # Compute P vals
            pvals[trait] = np.sum(storage[trait] > f) / n_preds

        # Create a DF from this
        ps = pd.DataFrame.from_dict(pvals, orient='index', columns=['P'])

        # Join with original, computed data
        analysed = pd.merge(left=original_data, right=ps, left_index=True, right_index=True)

        return analysed


    def _sequential_fit(self, predictors, y_array):
        """Carry out multivariate fitting with permutations for a given facet of data.
        """

        # Dictionary for storage
        model_pts = {}

        # Compute base_model, with just intercept + first predictor
        base_model = self._least_squares(predictors.iloc[:,:2], y_array)

        # Unpack the initial base model fit data
        model_pts[predictors.columns[1]] = [base_model[key] for key in ['f_ratio', 'r2']]

        # Start building up sequential fit, starting from constant + first trait
        for trait in predictors.columns[2:]:

            # Slice
            iter_preds = predictors.loc[:, 'const':trait]

            # Fit a new model with an extra predictor
            new_model = self._least_squares(iter_preds, y_array)

            # Include fit change stats
            change = self._fit_change(base_model, new_model)

            # Update base_model
            base_model = new_model

            # Store data
            model_pts[trait] = change

        # Create a dataframe with this data
        fit_data = pd.DataFrame.from_dict(model_pts, orient='index', columns=['F Ratio', 'R2', 'Diff Df Resid'])

        return fit_data


    def _least_squares(self, X, y):
        """Carries out least squares solution and returns regression metrics"""

        # Compute DF model and residual
        df_model = X.shape[1] - 1 # N-predictors including intercept
        no_constant = X.drop('const', 'columns')
        df_resid = no_constant.shape[0] - no_constant.shape[1] - 1

        # Make NumPy arrays
        X_ = X.values

        # Compute coefficients
        coefs, _, _, _ = np.linalg.lstsq(X_, y, rcond=False)

        # Fit with matrix multiplication
        y_pred = X_ @ coefs

        # Compute residuals
        residuals = y - y_pred

        # Compute the traces of the inner products of matrices for multivar equivalents to regression SS
        sst = np.trace(y.T @ y)
        ssm = np.trace(y_pred.T @ y_pred)
        ssr = np.trace(residuals.T @ residuals)

#        # This is equivalent to squaring matrix and summing all elements, possibly faster than np.trace
#         sst = (y ** 2).sum()
#         ssm = (y_pred ** 2).sum()
#         ssr = (residuals ** 2).sum()

        r2 = ssm / sst
        f_ratio = (ssm/df_model) / (ssr/df_resid)

        # Return regression metrics in dictionary
        metrics = {'SST': sst, 'SSM': ssm, 'SSR': ssr, 'r2': r2, 'f_ratio': f_ratio, 'df_model': df_model, 'df_resid': df_resid}

        return metrics

    def _fit_change(self, mod1, mod2):
        """Compute changes between models for F"""

        # Compute residual difference
        df_resid_diff = mod1['df_resid'] - mod2['df_resid']

        # Change in R2
        r2_change = mod2['r2'] - mod1['r2']

        # Change in F
        change_f = (mod1['SSR'] - mod2['SSR']) / df_resid_diff / mod2['SSR'] * mod2['df_resid']

        return change_f, r2_change, df_resid_diff
