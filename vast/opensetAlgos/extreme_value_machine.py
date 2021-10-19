"""
[Insert License Here]
=====

Modified from Derek's original 2020 wrapper for MultipleEVM class from VAST.
This also performs some modifications to the older MultipleEVM code for saving
and loading the EVM1vsRest objects.
"""
import logging
from dataclasses import dataclass

import h5py
import numpy as np
import torch

from exputils.data.labels import NominalDataEncoder
from exputils.ml.generic_predictors import SupervisedClassifier
from exputils.io import create_filepath

from vast.opensetAlgos.EVM import EVM_Training, EVM_Inference
from vast.DistributionModels.weibull import weibull


@dataclass
class EVM1vsRest(object):
    """A single 1 vs Rest classifier for a known class in the EVM model. This
    class is not intended to be used on its own during inference time, as it is
    only a part of the rest of the EVM. As such, this excludes the
    hyperparameters of the EVM to avoid redundancy and serves mostly as a
    struct with save and load methods.

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
    """

    # TODO consider saving EVM attribs in the EVM1vsRest. Redundant though.
    extreme_vectors: torch.Tensor
    extreme_vectors_indices: torch.Tensor
    weibulls: weibull

    # def __dict__(self):
    #    """Dictionary version of these attributes."""
    #    raise NotImplementedError()
    #    return

    def predict(self, points):
        raise NotImplementedError(
            "This is not necessary atm, but is techinically possible to do."
        )
        return

    def fit(self, points):
        raise NotImplementedError(
            "This is not necessary atm, but is techinically possible to do."
        )

    def save(self, h5, overwrite=False):
        """Save the model within the given HDF5 file."""
        # Open file for writing; create if not existent and avoid overwriting.
        if isinstance(h5, str):
            h5 = h5py.File(create_filepath(h5, overwrite), "w")

        # Save extreme vectors
        h5.create_dataset("extreme_vectors", data=self.extreme_vectors.cpu())

        # Save extreme vectors indices
        h5.create_dataset(
            "extreme_vectors_indices",
            data=self.extreme_vectors_indices.cpu(),
        )

        # Save weibulls
        h5_weibulls = h5.create_group("weibulls")
        params = self.weibulls.return_all_parameters()

        h5_weibulls.create_dataset("Scale", data=params["Scale"].cpu())
        h5_weibulls.create_dataset("Shape", data=params["Shape"].cpu())
        h5_weibulls.create_dataset("signTensor", data=params["signTensor"])
        # TODO rm translateAmountTensor as it is depract in current weibull
        h5_weibulls.create_dataset(
            "translateAmountTensor",
            data=params["translateAmountTensor"],
        )
        h5_weibulls.create_dataset(
            "smallScore",
            data=params["smallScoreTensor"].cpu(),
        )

    @staticmethod
    def load(h5):
        """Load the model from the given HDF5 file."""
        # Open the HDF5 for reading
        if isinstance(h5, str):
            h5 = h5py.File(h5, "r")

        # TODO rm translateAmountTensor as it is depract in current weibull
        h5_weibulls = h5["weibulls"]

        # Load extreme_vectors, extreme_vectors_indices, and weibulls
        return EVM1vsRest(
            torch.tensor(h5["extreme_vectors"][()]),
            torch.tensor(h5["extreme_vectors_indices"][()]),
            weibull(
                {
                    "Scale": torch.from_numpy(h5_weibulls["Scale"][()]),
                    "Shape": torch.from_numpy(h5_weibulls["Shape"][()]),
                    "signTensor": h5_weibulls["signTensor"][()],
                    "translateAmountTensor": h5_weibulls["translateAmountTensor"][()],
                    "smallScoreTensor": torch.from_numpy(h5_weibulls["smallScore"][()]),
                }
            ),
        )


@dataclass
class EVMParams(object):
    tailsize: float
    cover_threshold: float
    distance_multiplier: float
    distance_metric: str = "cosine"
    chunk_size: int = 200
    # TODO note the EVM_funcs expect these all to be in lists... Or just tailsize..


class ExtremeValueMachine(SupervisedClassifier):
    """Object Oriented Programming wrapper for the existing EVM funtions. This
    provides an object that contains a single EVM instance and streamlines the
    basic incremental supervised learning methods. The EVM consists of a
    1-vs-Rest classifier per known class and this keeps them and their
    internals together.

    Attributes
    ----------
    tail_size : int | float
        When an `int`, the number of distances on which the weibull models are
        fit. When a float, then it is the ratio of points in the initial fit to
        use as the tail size.
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

    Notes
    -----
    This class and EVM1vsRest are an initial OOP wrapper for EVM_Training and
    EVM_Inference such that the model is more easily managed. It may be
    possible for this to be more efficient by being written more directly using
    the PyTorch API. Adding the use of Ray where noted may aid the parallel
    processing as well.
    """

    def __init__(
        self,
        tail_size,
        cover_threshold,
        distance_multiplier,
        labels,
        distance_metric="cosine",
        chunk_size=200,
        device="cuda",
        tail_size_is_ratio=True,
        *args,
        **kwargs,
    ):
        # Create a NominalDataEncoder to handle the class encodings
        super(ExtremeValueMachine, self).__init__(labels, *args, **kwargs)
        self.one_vs_rests = None
        self._increments = 0
        self.device = torch.device(device)

        # TODO replace hotfix with upstream change for support for torch.device
        if self.device.index is None:
            raise ValueError("Upstream `vast` only supports indexed cuda torch devices")

        self.tail_size = tail_size
        if tail_size_is_ratio:
            self.tail_size_int = None
        else:
            self.tail_size_int = self.tail_size

        self.cover_threshold = cover_threshold
        self.distance_multiplier = distance_multiplier
        self.distance_metric = distance_metric
        self.chunk_size = chunk_size

    @property
    def _args(self):
        """Property that obtains the `args` of the EVM as expected for input
        into the vast.opensetAlgos.EVM.EVM_Train and EVM_Inference functions.

        Returns
        -------
        dict()
            A dictionary of the EVM's hyperparameters as used by vast package.
        """
        # NOTE EVM_funcs expect iterables for all these...
        return EVMParams(
            [self.tail_size_int],
            [self.cover_threshold],
            [self.distance_multiplier],
            self.distance_metric,
            self.chunk_size,
        )

    @property
    def get_increment(self):
        return self._increments

    # TODO make this a ray function for easy parallelization.
    def fit(self, points, labels=None, extra_negatives=None, init_fit=None):
        """Fit the model with the given data either as initial fit or increment.
        Defaults to incremental fitting.

        Args
        ----
        points : torch.Tensor | [torch.Tensor]
        labels : torch.Tensor | [str | int] = None
            The encoded labels as integers or a list of the unencoded labels
            that corresponds to the list of pytorch tensors in `points`.
        extra_negatives : torch.Tensor = None
            Data points that will always serve as negative class samples for
            ever EVM1vsRest. These are treated as unlabeled or unknown classes,
            though may be classes that are known but just are not cared about
            to be classified as known for the given classification task. For
            example, for a classifier of dog breedss, one may happen to have
            images of different cat breeds and even have their labels, but
            their task is to classify dogs and detect when others are not dogs,
            so all cat images are treated as extra negatives.
        init_fit : bool = None
            By default, the ExtremeValueMachine keeps track of the number of
            increments.
        """
        # If points and labels are aligned sequence pair (X, y): adjust form
        if (
            isinstance(points, np.ndarray)
            and (isinstance(labels, list) or isinstance(labels, np.ndarray))
            and len(points) == len(labels)
        ):
            # Adjust sequence pair into list of torch.Tensors and unique labels
            # TODO handle np and torch versions.
            unique = np.unique(labels)
            labels = np.array(labels)
            points = [torch.Tensor(points[labels == u]) for u in unique]
            labels = unique
        elif (
            isinstance(points, torch.Tensor)
            and (isinstance(labels, list) or isinstance(labels, torch.Tensor))
            and len(points) == len(labels)
        ):
            labels = torch.Tensor(labels)
            unique = torch.unique(labels)
            points = [points[labels == u] for u in unique]
        elif isinstance(points, list):
            if all([isinstance(pts, np.ndarray) for pts in points]):
                # If list of np.ndarrays, turn into torch.Tensors
                points = [torch.Tensor(pts) for pts in points]
            elif not all([isinstance(pts, torch.Tensor) for pts in points]):
                raise TypeError(
                    " ".join(
                        [
                            "expected points to be of types: list(np.ndarray),",
                            "list(torch.tensor), or np.ndarray with labels as an",
                            "aligned list or np.ndarray",
                        ]
                    )
                )
        else:
            raise TypeError(
                " ".join(
                    [
                        "expected points to be of types: list(np.ndarray),",
                        "list(torch.Tensor), or (np.ndarray torch.Tensor) with labels",
                        "as an aligned (list or np.ndarray)",
                    ]
                )
            )

        # Ensure extra_negatives is of expected form (no labels for these)
        if (
            isinstance(extra_negatives, np.ndarray) and len(extra_negatives.shape) == 2
        ) or isinstance(extra_negatives, list):
            extra_negatives = torch.Tensor(extra_negatives)
        elif extra_negatives is not None and not (
            isinstance(extra_negatives, torch.Tensor) and len(extra_negatives.shape) == 2
        ):
            raise TypeError(
                " ".join(
                    [
                        "The extra_negatives must be either None, torch.Tensor of",
                        "shape 2, or an object broadcastable to such a torch.Tensor.",
                        f"But recieved type `{type(extra_negatives)}`.",
                    ]
                )
            )

        if init_fit or (init_fit is None and self._increments == 0):
            self._initial_fit(points, extra_negatives)
        else:  # Incremental fit
            self._increment_fit(points, extra_negatives)

    # TODO make this a ray function for easy parallelization.
    def _initial_fit(self, points, extra_negatives=None):
        """Fits the EVM with the given points as if it is the first fitting.

        Args
        ----
        points : torch.Tensor | [torch.Tensor]
        labels : torch.Tensor | [str | int] = None
        extra_negatives : torch.Tensor = None
        """
        if self.tail_size_int is None:
            self.tail_size_int = int(np.round(self.tail_size * len(points)))

        evm_fit = EVM_Training(
            list(self.label_enc.encoder.inv),
            {i: pts for i, pts in enumerate(points)},
            self._args,
            self.device.index,
        )

        self.one_vs_rests = {
            one_vs_rest[1][0]: EVM1vsRest(
                one_vs_rest[1][1]["extreme_vectors"],
                one_vs_rest[1][1]["extreme_vectors_indexes"],
                one_vs_rest[1][1]["weibulls"],
            )
            for one_vs_rest in evm_fit
        }
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
        # TODO figure out how to efficiently update a subset of the EVM1vsRests
        # Same thing as initial fit but it should be more flexible.
        evm_fit = EVM_Training(
            list(self.label_enc.encoder.inv),
            {i: pts for i, pts in enumerate(points)},
            self._args,
            self.device.index,
            {k: vars(v) for k, v in self.one_vs_rests.items()},
        )

        self.one_vs_rests = {
            one_vs_rest[1][0]: EVM1vsRest(
                one_vs_rest[1][1]["extreme_vectors"],
                one_vs_rest[1][1]["extreme_vectors_indexes"],
                one_vs_rest[1][1]["weibulls"],
            )
            for one_vs_rest in evm_fit
        }
        self._increments += 1

    # TODO make this a ray function for easy parallelization. Lesser priority
    def one_vs_rest_probs(self, points, gpu=None):
        """Predicts the 1 vs Rest class probabilities for the points. This
        vector does not sum to one!
        """
        if not isinstance(points, torch.Tensor):
            raise TypeError("expected `points` to be of type: torch.Tensor")

        # TODO figure out how to make use of the efficient inference function.
        # TODO this currently only uses 1 batch, rather than efficient or user
        # defined bathces. So this need changed but it should run for now.
        # pos_cls_name in EVM_Inference is actually the batch ID. So keys are
        # batch ids and values are the tensors of the different batches.
        return next(
            EVM_Inference(
                ["batch"],
                {"batch": points},
                self._args,
                self.device.index,
                # Create the models as expected by EVM_Inference
                {k: vars(v) for k, v in self.one_vs_rests.items()},
            )
        )[1][1]

    def known_probs(self, points, gpu=None):
        """Predicts known probability vector without the unknown class.
        Args
        ----
        points : torch.Tensor

        Returns
        -------
        torch.Tensor
            A torch tensor of a vector of probabilities per sample, including
            an unknown class probability as the last dimension.
        """
        probs = self.one_vs_rest_probs(points)
        return probs / probs.sum(1, True)

    # TODO make this a ray function for easy parallelization. Lesser priority
    def predict(self, points, unknown_last_dim=True):
        """Predicts the probability vector of knowns and unknown per sample.

        The resulting probability vector per sample is constructed such that
        all EVM1vsRest binary classification probabilities are concatenated
        into a vector and the unknown class probability is included, locaiton
        depending on `unknown_last_dim`. The unknown class probability is the
        inverse probability of the maximum known/positive probability from all
        EVM1vsRest classifiers.

        This unnormalized vector is then normalized by summing all of the
        values in that vector and dividing them by that sum, otherwise known as
        dividing the vector by L1 norm of that vector. This preserves the
        ratios of the probabilities to one another. Remember that the negative
        samples per EVM1vsRest includes the samples labeled as other classes,
        which makes the EVM1vsRest dependent binary classifiers.

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
        probs = self.one_vs_rest_probs(points)

        # Find probability of unknown as 1 - max 1-vs-Rest and concat
        if unknown_last_dim:
            probs = torch.cat((probs, 1 - torch.max(probs, 1, True).values), 1)
        else:
            probs = torch.cat((1 - torch.max(probs, 1, True).values, probs), 1)

        # Get a normalized probability vector keeping raitos of values.
        return probs / probs.sum(1, True)

    def save(self, h5, overwrite=False):
        """Saves the EVM model as HDF5 to disk with the labels and parameters."""
        if self.one_vs_rests is None:
            raise RuntimeError("The ExtremeValueMachine has not been trained yet.")

        # Open file for writing; create if not existent and avoid overwriting.
        if isinstance(h5, str):
            h5 = h5py.File(create_filepath(h5, overwrite), "w")

        # Save the EVMs
        for idx, one_vs_rest in self.one_vs_rests.items():
            one_vs_rest.save(h5.create_group(f"EVM1vsRest-{idx}"))

        # Save the labels for the encoder
        labels = self.labels
        h5.attrs["labels_dtype"] = str(labels.dtype)

        # NOTE This does not save the encoding of the class if diff from [0,
        # len(labels)]!!!!!! Order is preserved though.
        if labels.dtype.type is np.str_ or labels.dtype.type is np.string_:
            h5.create_dataset(
                "labels",
                data=labels.astype(object),
                dtype=h5py.special_dtype(vlen=str),
            )
        else:
            h5["labels"] = labels

        # Write hyperparameters
        for attrib in [
            "tail_size",
            "cover_threshold",
            "distance_metric",
            "distance_multiplier",
            "chunk_size",
            "_increments",
            "tail_size_int",
        ]:
            h5.attrs[attrib] = getattr(self, attrib)

    @staticmethod
    def load(
        h5,
        labels=None,
        labels_dtype=None,
        train_hyperparams=None,
        device="cuda",
    ):
        """Performs the same load functionality as in MultipleEVM but loads the
        ordered labels from the h5 file for the label encoder and other
        hyperparameters if they are present.
        """
        if isinstance(h5, str):
            h5 = h5py.File(h5, "r")

        # Load the ordered labels into the NominalDataEncoder
        if "labels" in h5.keys():
            if labels is not None:
                logging.info(
                    " ".join(
                        [
                            "`labels` key exists in the HDF5 MEVM state file, but",
                            "labels was given explicitly to `load()`. Ignoring the",
                            "labels in the HDF5 file.",
                        ]
                    )
                )
            else:
                if labels_dtype is None:
                    labels_dtype = np.dtype(h5.attrs["labels_dtype"])
                labels = h5["labels"][:].astype(labels_dtype)
        elif labels is None:
            raise KeyError(
                " ".join(
                    [
                        "No `labels` dataset available in given hdf5.",
                        "and `labels` parameter is None",
                    ]
                )
            )

        # Load the EVM1vsRest models
        one_vs_rests = {}
        for i, label in enumerate(labels):
            if f"EVM1vsRest-{i}" in h5.keys():
                one_vs_rests[i] = EVM1vsRest.load(h5[f"EVM1vsRest-{i}"])

        # Load training vars if not given
        if train_hyperparams is None:
            # NOTE Able to specify which to load from h5 by passing a list.
            train_hyperparams = [
                "tail_size",
                "cover_threshold",
                "distance_metric",
                "distance_multiplier",
                "chunk_size",
                "_increments",
                "tail_size_int",
            ]

        if isinstance(train_hyperparams, list):
            train_hyperparams = {
                attr: h5.attrs[attr] for attr in train_hyperparams if attr in h5.attrs
            }
        elif not isinstance(train_hyperparams, dict):
            raise TypeError(
                " ".join(
                    [
                        "`train_hyperparams` expected type: None, list, or dict, but",
                        f"recieved {type(train_hyperparams)}",
                    ]
                )
            )

        _increments = train_hyperparams.pop("_increments")
        tail_size_int = train_hyperparams.pop("tail_size_int")
        evm = ExtremeValueMachine(
            labels=labels,
            device=device,
            **train_hyperparams,
        )
        evm.one_vs_rests = one_vs_rests
        evm._increments = _increments
        evm.tail_size_int = tail_size_int

        return evm
