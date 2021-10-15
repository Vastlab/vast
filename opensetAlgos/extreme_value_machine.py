"""
[Insert License Here]
=====

Modified from Derek's original 2020 wrapper for MultipleEVM class from VAST.
This also performs some modifications to the older MultipleEVM code for saving
and loading the EVM1vsRest objects.
"""
import logging

import h5py
import numpy as np
import torch

from exputils.data.labels import NominalDataEncoder
from exputils.ml.generic_predictors import SupervisedClassifier
from exputils.io import create_filepath

#from vast.opensetAlgos.EVM import EVM_Training, EVM_Inference


class EVM1vsRest(object):
    """A single 1 vs Rest classifier for a known class in the EVM model. This
    class is not intended to be used on its own during inference time, as it is
    a only a part of the rest of the EVM. As such, this excludes the
    hyperparameters of the EVM.

    Attributes
    ----------
    extreme_vectors : torch.Tensor
        A vector of exemplar samples used by the EVM.
    extreme_vectors_indices : torch.LongTensor
        The index of the extreme_vectors, only useful in case of reducing the
        size of the EVM model saved.
    weibulls: vast.weibull.weibull
        A set of 1-D Weibull distributions fit to the distances and that
        correspond to each extreme vector.
        The output of `weibulls.return_all_parameters()` combined with the
        `extreme_vectors` is the EVM 1vsRest for one class (This class).
    class : int | str | None, optional
        String or int identifier of the class this EVM1vsRest belongs to.
    """
    def __init__(self):
        raise NotImplementedError()

    def predict(self, ):
        raise NotImplementedError(
            'This is not necessary atm, but is techinically possible to do.'
        )
        return

    def fit(self,):
        raise NotImplementedError(
            'This is not necessary atm, but is techinically possible to do.'
        )

    def save(self, h5):
        """Save the model within the given H5DF file."""
        raise NotImplementedError()

    @staticmethod
    def load(h5):
        """Load the model from the given H5DF file."""
        raise NotImplementedError()


class ExtremeValueMachine(SupervisedClassifier):
    """Object Oriented Programming wrapper for the existing EVM funtions. This
    provides an object that contains a single EVM instance and streamlines the
    basic incremental supervised learning methods. The EVM consists of a
    1-vs-Rest classifier per known class and this keeps them and their
    internals together.

    Attributes
    ----------
    tail_size : int | float | torch.Tensor.float
        When an `int`, the number of distances on which the weibull models are
        fit.
    cover_threshold : float | torch.Tensor.float
    distance_multiplier : float | Torch
    distance_metric : 'cosine' | 'euclidean'
        The distance metric to use either 'cosine' or 'euclidean'.
    chunk_size : int = 200
    one_vs_rests: {class_enc: EVM1vsRest} = None
        A dictionary of class encoding to that class' 1 vs Rest classifier.
        These 1 vs Rest classifiers form this EVM model.
    label_enc : NominalDataEncoder
        The encoder to manage the labels known to the EVM.
    _increments : int = 0
        The number of incremental learning phases completed.
    device : torch.device = 'cuda'
        The torch device to compute fitting and predictions on.

    max_unknowns : int = None
        The total number of unknowns expected by the EVM, used when performing
        incremental updates (currently not implemented in EVM).
    detection_threshold : float = None
        The threshold to apply to each sample's probability vector as an ad hoc
        solution to adjusting the MEVM's sensitivity to unknowns.
    """
    def __init__(
        self,
        tail_size,
        cover_threshold,
        distance_multiplier,
        labels,
        distance_metric='cosine',
        chunk_size=200,
        device='cuda',
        *args,
        **kwargs,
    ):
        # Create a NominalDataEncoder to handle the class encodings
        super(ExtremeValueMachine, self).__init__(labels, *args, **kwargs)
        self.one_vs_rests = None
        self._increment = 0
        self.device = torch.device(device)

    # TODO make this a ray function for easy parallelization.
    def fit(self, points, labels=None,  extra_negatives=None, init_fit=None):
        """Fit the model with the given data either as initial fit or increment.
        Defaults to incremental fitting.

        Args
        ----
        points : torch.Tensor | [torch.Tensor]
        labels : torch.Tensor | [str | int] = None
            The encoded labels as integers or a list of the unencoded labels
            that corresponds to the list of pytorch tensors in `points`.
        extra_negatives : torch.Tensor = None
        init_fit : bool = None
            By default, the ExtremeValueMachine keeps track of the number of
            increments.
        """

        # TODO point and label conversion

        # If points and labels are aligned sequence pair (X, y): adjust form
        if (
            isinstance(points, np.ndarray)
            and (isinstance(labels, list) or isinstance(labels, np.ndarray))
            and len(points) == len(labels)
        ):
            # Adjust sequence pair into list of torch.Tensors and unique labels
            unique = np.unique(labels)
            labels = np.array(labels)
            points = [torch.Tensor(points[labels == u]) for u in unique]
            labels = unique
        elif isinstance(points, list):
            if all([isinstance(pts, np.ndarray) for pts in points]):
                # If list of np.ndarrays, turn into torch.Tensors
                points = [torch.Tensor(pts) for pts in points]
            elif not all([isinstance(pts, torch.Tensor) for pts in points]):
                raise TypeError(' '.join([
                    'expected points to be of types: list(np.ndarray),',
                    'list(torch.tensor), or np.ndarray with labels as an',
                    'aligned list or np.ndarray',
                ]))
        else:
            raise TypeError(' '.join([
                'expected points to be of types: list(np.ndarray),',
                'list(torch.Tensor), or (np.ndarray torch.Tensor) with labels',
                'as an aligned (list or np.ndarray)',
            ]))

        # Ensure extra_negatives is of expected form (no labels for these)
        if (
            (
                isinstance(extra_negatives, np.ndarray)
                and len(extra_negatives.shape) == 2
            )
            or isinstance(extra_negatives, list)
        ):
            extra_negatives = torch.Tensor(extra_negatives)
        elif not (
            isinstance(extra_negatives, torch.Tensor)
            and len(extra_negatives.shape) == 2
        ):
            raise TypeError(' '.join([
                'The extra_negatives must be either None, torch.Tensor of',
                'shape 2, or an object broadcastable to such a torch.Tensor.',
                f'But recieved type `{type(extra_negatives)}`.',
            ]))

        if init_fit or (init_fit is None and self._increment == 0):
            self._initial_fit(points, extra_negatives)
        else: # Incremental fit
            self._increment_fit(points, extra_negatives)

    # TODO make this a ray function for easy parallelization.
    def _initial_fit(self, points, extra_negatives=None):
        """Fits the

        Args
        ----
        points : torch.Tensor | [torch.Tensor]
        labels : torch.Tensor | [str | int] = None
        extra_negatives : torch.Tensor = None
        """
        EVM_Training(
            list(self.label_enc.encoder.inv),
            points,
            args,
            self.device,
            models,
        )

        #self.one_vs_rests =
        self._increments = 1

    # TODO make this a ray function for easy parallelization.
    def _increment_fit(self, points, extra_negatives=None):
        """Performs and incremental fitting of the current EVM.
        Args
        ----
        points : torch.Tensor | [torch.Tensor]
        labels : torch.Tensor | [str | int] = None
        extra_negatives : torch.Tensor = None
        """
        #self.one_vs_rests =
        self._increments += 1

    # TODO make this a ray function for easy parallelization. Lesser priority
    def one_vs_rest_probs(self, points, gpu=None):
        """Predicts the 1 vs Rest class probabilities for the points. This
        vector does not sum to one!
        """
        return EVM_Inference(
                list(self.label_enc.encoder.inv),
                points,
                args,
                self.device,
                self.one_vs_rests,
            )[1][1]

    def known_probs(self, points, gpu=None):
        """Predicts known probability vector without the unknown class."""
        probs = self.one_vs_rest_probs(points)
        return probs / probs.sum(1, True)

    # TODO make this a ray function for easy parallelization. Lesser priority
    def predict(self, points, unknown_last_dim=True):
        """Wraps the MultipleEVM's class_probabilities and uses the encoder to
        keep labels as expected by the user. Also adjusts the class
        probabilities to include the unknown class.

        Args
        ----
        points : torch.Tensor
        unknown_last_dim : bool = True
            If True, the element of the probability vector representing the
            unknown class is appeneded to the end of the vector. Otherwise, it
            is at the beginning.

        Returns
        -------
        torch.Tensor
            A torch tensor of a vector of probabilities per sample, including
            an unknown class probability as the last dimension.
        """
        if not isinstance(points, torch.Tensor):
            raise TypeError('expected `points` to be of type: torch.Tensor')

        probs = self.one_vs_rest_probs(points)

        # Find probability of unknown as 1 - max 1-vs-Rest and concat
        if unknown_last_dim:
            probs = torch.cat((probs, 1 - torch.max(probs, 1, True).values), 1)
        else:
            probs = torch.cat((1 - torch.max(probs, 1, True).values, probs), 1)

        # Get a normalized probability vector keeping raitos of values.
        return probs / probs.sum(1, True)

    def save(self, h5, overwrite=False):
        """Saves the EVM model as H5DF to disk with the labels and parameters.
        """
        if self.one_vs_rests is None:
            raise RuntimeError("The model has not been trained yet.")

        # Open file for writing; create if not existent and avoid overwriting.
        if isinstance(h5, str):
            h5 = h5py.File(create_filepath(h5, overwrite), 'w')

        # Save the EVMs
        for idx, one_vs_rest in self.one_vs_rests.items():
            one_vs_rest.save(h5.create_group(f'EVM1vsRest-{idx}'))

        # Save the labels for the encoder
        labels = self.labels
        h5.attrs['labels_dtype'] = str(labels.dtype)

        # NOTE This does not save the encoding of the class if diff from [0,
        # len(labels)]!!!!!! Order is preserved though.
        if labels.dtype.type is np.str_ or labels.dtype.type is np.string_:
            h5.create_dataset(
                'labels',
                data=labels.astype(object),
                dtype=h5py.special_dtype(vlen=str),
            )
        else:
            h5['labels'] = labels

        # Write hyperparameters
        for attrib in [
            'tail_size',
            'cover_threshold',
            'distance_function',
            'distance_multiplier',
            'chunk_size',
            '_increments',
        ]:
            h5.attrs[attrib] = getattr(self, attrib)

    @staticmethod
    def load(h5, labels=None, labels_dtype=None, train_hyperparams=None):
        """Performs the same load functionality as in MultipleEVM but loads the
        ordered labels from the h5 file for the label encoder and other
        hyperparameters if they are present.
        """
        if isinstance(h5, str):
            h5 = h5py.File(h5, 'r')

        # Load the ordered labels into the NominalDataEncoder
        if 'labels' in h5.keys():
            if labels is not None:
                logging.info(' '.join([
                    '`labels` key exists in the HDF5 MEVM state file, but',
                    'labels was given explicitly to `load()`. Ignoring the',
                    'labels in the HDF5 file.',
                ]))
            else:
                if labels_dtype is None:
                    labels_dtype = np.dtype(h5.attrs['labels_dtype'])
                labels = h5['labels'][:].astype(labels_dtype)
        elif labels is None:
            raise KeyError(' '.join([
                'No `labels` dataset available in given hdf5.',
                'and `labels` parameter is None',
            ]))

        # Load the EVM1vsRest models
        one_vs_rests = {}
        for i, label in enumerate(labels):
            if f'EVM1vsRest-{i}' in h5.keys():
                one_vs_rests[i] = EVM1vsRest.load(h5[f'EVM1vsRest-{i}'])

        # Load training vars if not given
        if train_hyperparams is None:
            # NOTE Able to specify which to load from h5 by passing a list.
            train_hyperparams = [
                'tail_size',
                'cover_threshold',
                'distance_function',
                'distance_multiplier',
                'chunk_size',
                '_increments',
            ]

        if isinstance(train_hyperparams, list):
            train_hyperparams = {
                attr: h5.attrs[attr] for attr in train_hyperparams
                if attr in h5.attrs
            }
        elif not isinstance(train_hyperparams, dict):
            raise TypeError(' '.join([
                '`train_hyperparams` expected type: None, list, or dict, but',
                f'recieved {type(train_hyperparams)}',
            ]))

        _increments = train_hyperparams.pop('_increments')
        evm = ExtremeValueMachine(labels=labels, **train_hyperparams)
        evm.one_vs_rests = one_vs_rests
        evm._increments = _increments

        return evm
