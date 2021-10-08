"""
[Insert License Here]
=====

Modified from Derek's original wrapper for MultipleEVM class from 2020.
"""
import logging

import h5py
import numpy as np
import torch

from exputils.data.labels import NominalDataEncoder
from exputils.ml.generics import SupervisedClassifier

from vast.


class EVM1vsRest(object):
    """A single 1 vs Rest classifier for a known class in the EVM."""
    def __init__(self):
        raise NotImplementedError()


class ExtremeValueMachine(object):
    """Object Oriented Programming wrapper for the existing EVM funtions. This
    provides an object that contains a single EVM instance and streamlines the
    basic incremental supervised learning methods. The EVM consists of a
    1-vs-Rest classifier per known class and this keeps them and their
    internals together.

    Attributes
    ----------
    tailsize : int | float | torch.Tensor.float
    cover_threshold : float | torch.Tensor.float
    distance_multiplier : float | Torch
    distance_metric : str | func = 'cosine'
    chunk_size : int = 200

    extreme_vectors:
    one_vs_rest_extreme_classifer:
    label_enc : NominalDataEncoder
        The encoder to manage the labels known to the MEVM.

    max_unknowns : int = None
        The total number of unknowns expected by the MEVM, used when performing
        incremental updates (currently not implemented in MEVM).
    detection_threshold : float = None
        The threshold to apply to each sample's probability vector as an ad hoc
        solution to adjusting the MEVM's sensitivity to unknowns.
    """
    def __init__(
        self,
        tailsize,
        cover_threshold,
        distance_multiplier,
        distance_metric='cosine',
        chunk_size=200,
        labels=None,
        #max_unknown=None,
        #detection_threshold=None,
    ):
        #self.max_unknown = max_unknown
        #self.detection_threshold = detection_threshold

        super(ExtremeValueMachine, self).__init__(*args, **kwargs)

        # Create a NominalDataEncoder to map class inputs to the MEVM internal
        # class represntation.
        if isinstance(labels, NominalDataEncoder) or labels is None:
            self.label_enc = labels
        elif isinstance(labels, list) or isinstance(labels, np.ndarray):
            self.label_enc = NominalDataEncoder(labels)
        else:
            raise TypeError(' '.join([
                'Expected `labels` of types: None, list, np.ndarray, or',
                'NominalDataEncoder, not of type {type(labels)}'
            ]))

    def save(self, path):
        """Save the model to the given location.
        """

    @staticmethod
    def load(self, path):
        """Load the model from the given location.
        """

    def fit(self, init_fit=False):
        """Fit the model with the given data either as initial fit or increment.
        Defaults to incremental fitting.

        Args
        ----
        input_samples
        labels
        init_fit : bool = False
        """

    def _initial_fit(self, ):
        """
        Args
        ----
        input_samples
        labels
        """

    def _increment_fit(self, ):
        """
        Args
        ----
        input_samples
        labels
        """

    def save(self, h5):
        """Performs the same save functionality as in MultipleEVM but adds a
        dataset for the encoder's ordered labels and also saves the
        hyperparameters for ease of storing and loading.
        """
        if self._evms is None:
            raise RuntimeError("The model has not been trained yet.")

        # Open file for writing; create if not existent
        if isinstance(h5, str):
            h5 = h5py.File(h5, 'w')

        # Write EVMs
        for i, evm in enumerate(self._evms):
            evm.save(h5.create_group("EVM-%d" % (i+1)))

        # Write labels for the encoder
        if self.label_enc is None:
            logging.info('No labels to be saved.')
        else:
            labels = self.labels
            h5.attrs['labels_dtype'] = str(labels.dtype)

            if labels.dtype.type is np.str_ or labels.dtype.type is np.string_:
                h5.create_dataset(
                    'labels',
                    data=labels.astype(object),
                    dtype=h5py.special_dtype(vlen=str),
                )
            else:
                h5['labels'] = labels

        # Write training vars
        for attrib in ['tailsize', 'cover_threshold', 'distance_function',
            'distance_multiplier', 'max_unknown', 'detection_threshold',
        ]:
            value = getattr(self, attrib)
            if value is not None:
                h5.attrs[attrib] = value

    @staticmethod
    def load(h5, labels=None, labels_dtype=None, train_hyperparams=None):
        """Performs the same load functionality as in MultipleEVM but loads the
        ordered labels from the h5 file for the label encoder and other
        hyperparameters if they are present.
        """
        if isinstance(h5, str):
            h5 = h5py.File(h5, 'r')

        # load evms
        _evms = []
        i = 1
        while "EVM-%d" % i in h5:
            _evms.append(EVM(h5["EVM-%d" % (i)], log_level='debug'))
            i += 1

        # Load the ordered label into the NominalDataEncoder
        if 'labels' in h5.keys():
            if labels is not None:
                logging.info(' '.join([
                    '`labels` key exists in the HDF5 MEVM state file, but',
                    'labels was given explicitly to MEVM.load(). Ignoring the',
                    'labels in the HDF5 file.',
                ]))
                label_enc = NominalDataEncoder(labels)
            else:
                if labels_dtype is None:
                    labels_dtype = np.dtype(h5.attrs['labels_dtype'])
                label_enc = NominalDataEncoder(
                    h5['labels'][:].astype(labels_dtype),
                )
        elif labels is not None:
            label_enc = NominalDataEncoder(labels)
        else:
            logging.warning(' '.join([
                'No `labels` dataset available in given hdf5. Relying on the',
                'evm model\'s labels if they exist. Will fail if the MEVM',
                'state does not have any labels in each of its EVM.',
            ]))

            label_enc = NominalDataEncoder(
                [evm.label for evm in _evms],
            )

        # Load training vars if not given
        if train_hyperparams is None:
            # NOTE Able to specify which to load from h5 by passing a list.
            train_hyperparams = [
                'tailsize',
                'cover_threshold',
                'distance_function',
                'distance_multiplier',
                'max_unknown',
                'detection_threshold',
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

        mevm = MEVM(label_enc, **train_hyperparams)
        mevm._evms = _evms

        return mevm

    #def train(self, *args, **kwargs):
    #    # NOTE this may be necessary if train or train_update are used instead
    #    # of fit to keep the encoder in sync!
    #    super(MEVM, self).train(*args, **kwargs)
    #    self.label_enc = NominalDataEncoder([evm.label for evm in self._evms])

    def fit(self, points, labels=None, extra_negatives=None):
        """Wraps the MultipleEVM's train() and uses the encoder to
        """
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
                'list(torch.tensor), or np.ndarray with labels as an',
                'aligned list or np.ndarray',
            ]))

        # Set encoder if labels is not None
        if labels is not None:
            if len(points) != len(labels):
                raise ValueError(' '.join([
                    'The given number of labels does not equal the number of',
                    'classes represented by the list of points.',
                    'If giving an aligned sequence pair of points and labels,',
                    'then ensure `points` is of type `np.ndarray`.',
                ]))

            if self.label_enc is not None:
                logging.debug(
                    '`encoder` is not None and is being overwritten!',
                )

            if isinstance(labels, NominalDataEncoder):
                self.label_enc = labels
            elif isinstance(labels, list) or isinstance(labels, np.ndarray):
                self.label_enc = NominalDataEncoder(labels)
            else:
                raise TypeError(' '.join([
                    'Expected `labels` of types: None, list, np.ndarray, or',
                    'NominalDataEncoder, not of type {type(labels)}'
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
        elif not isinstance(extra_negatives, torch.Tensor):
            raise TypeError(' '.join([
                'The extra_negatives must be either None, torch.Tensor of',
                'shape 2, or an object broadcastable to such a torch.Tensor.',
                f'But recieved type `{type(extra_negatives)}`.',
            ]))

        # Points is now list(torch.Tensors) and encoder handled.

        # TODO handle adjust of extra negatives as a list of labels to be known
        # unknowns. For now, expects extra_negatives always of correct type.
        self.train(points, labels, extra_negatives)

    def predict(self, points, return_tensor=False, threshold_unknowns=False):
        """Wraps the MultipleEVM's class_probabilities and uses the encoder to
        keep labels as expected by the user. Also adjusts the class
        probabilities to include the unknown class.

        Args
        ----
        points : torch.Tensor
        return_tesnor : bool = False
        threshold_unknowns : bool = False
            If True, applies the unknown threshold to the probability vector
            and returns the resulting argmax of the probability vector per
            sample.

        Returns
        -------
        np.ndarray
        """
        if isinstance(points, np.ndarray):
            points = torch.Tensor(points)
        elif not isinstance(points, torch.Tensor):
            raise TypeError(
                'expected points to be of type: np.ndarray or torch.Tensor',
            )

        probs = self.class_probabilities(points)

        if return_tensor:
            raise NotImplementedError('Not yet.')

        # Find probability of unknown as its own class
        probs = np.array(probs)
        max_probs_known = probs.max(axis=1).reshape(-1, 1)
        unknown_probs = 1 - max_probs_known

        # Scale the rest of the known class probs by max prob known
        probs *= max_probs_known
        probs = np.hstack((probs, unknown_probs))

        # TODO if threshold_unknowns: Apply thresholding
        if threshold_unknowns:
            if self.detection_threshold is None:
                raise ValueError('`detection_threshold` is not set!')
            argmax = probs.argmax(1)
            argmax[
                probs[np.arange(probs.shape[0]), argmax] < threshold[0]
            ] = self.labels.unknown_idx

            return argmax

        return probs
