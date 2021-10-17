"""
[Insert License here]

=====

This is a simple automated test of the extreme value machine wrapper to
ensure it functions as desired. The toy task is a simple simulation of four 2D
gaussians whose locations are centered on the edge of the unit circle of this
space, along the axes of this space, that is centered at coordinates [0,1],
[1,0], [0,-1], and [-1,0]. These gaussians form the distributions of samples
from four different classes to be classified. The classification task is given
the 2D coordinates, classify that point with a probability vector to specify
which class it most likely belongs. Naturally, being evenly spaced from one
another on the unit circle in this 2D space, the coordinates near [0,0] are the
most uncertain and are expected to produce the closed classification
probability vector of [0.25, 0.25, 0.25, 0.25] at coordinates [0,0]. This toy
problem allows for basic assessment of the ExtremeValueMachine class to ensure
it can at least fit, predict, save, and load etc., and possibly be used for
diagnosis of performance of the EVM.
"""
import pytest
import torch

from vast.opensetAlgos.extreme_value_machine import ExtremeValueMachine


class ToyClassify2D4MVNs(object):
    """Create the generative sampling procedure for obtaining coordinates of
    points as feature data along with the label of which Gaussian distribution
    they belong to. The Gaussian distributions are labeled their index which
    starts at zero at the top most Gaussian centered at [0, 1] and labels the
    rest that follow clockwise around the unit circle.
    """
    def __init__(self, locs=None, scales=0.5, labels=None, seed=None):
        # TODO create PyTorch Gaussian Distributions at locs and scales
        if locs is None:
            locs =  [[1, 0], [0, 1], [-1, 0], [0, -1]]

        if not isinstance(scales, list):
            scales = [scales] * len(locs)

        self.mvns = [
            torch.distributions.multivariate_normal.MultivariateNormal(
                torch.Tensor(loc),
                torch.eye(2) * scales[i],
            )
            for i, loc in enumerate(locs)
        ]

    def eq_sample_n(self, num, randperm=True):
        if randperm:
            idx = torch.randperm(num * len(self.mvns))
            return (
                torch.cat([
                    mvn.sample_n(num) for mvn in self.mvns
                ])[idx],
                torch.Tensor([i] * num for i in range(len(self.mvns)))[idx]
            )
        return (
            torch.cat([
                mvn.sample_n(num) for mvn in self.mvns
            ]),
            torch.Tensor([i] * num for i in range(len(self.mvns))),
        )


class TestEVMToyClassify2D4MVNs(object):
    """PyTest unit tests using the toy classification simulation.

    Variables
    ---------
    toy_sim : ToyClassify2D4MVNs
        The toy simulated classification problem.
    evm : ExtremeValueMachine
        The EVM to be tested.
    train_features : torch.Tensor
        The train coordinates to be classified.
    train_labels : torch.Tensor
        The train coordinates' corresponding labels.
    train_features : torch.Tensor
        The test coordinates to be classified.
    train_labels : torch.Tensor
        The test coordinates' corresponding labels.
    """
    def setup(self, seed=0, device='cuda:0'):
        """The setup for all tests in this class."""
        # Create toy simulation
        toy_sim = ToyClassify2D4MVNs(seed=seed)

        # Create ExtremeValueMachine
        evm = ExtremeValueMachine(
            10,
            0.5,
            1.0,
            [0, 1, 2, 3],
            'cosine',
            device=device,
            tail_size_is_ratio=False,
        )

        return toy_sim, evm

    def test_extreme_value_machine_no_unknowns(
        self,
        tmp_path='/tmp/vast/extreme_value_machine/evm.h5',
        train_num_each=10,
        test_num_each=30,
        inc_train_num_each=100,
        inc_test_num_each=100,
        seed=0,
        device='cuda:0',
    ):
        """Tests the ExtremeValueMachine for basic functionality."""
        toy_sim, evm = self.setup(seed, device)

        with pytest.raises(RuntimeError):
            # Confirm it throws an error when trying to save unfit EVM.
            evm.save(tmp_path)

        # Generate train samples
        train_features, train_labels = toy_sim.eq_sample_n(train_num_each)

        # Check increment value == 0
        assert evm.get_increment == 0

        evm.fit(train_features, train_labels)

        # Check increment value == 1
        assert evm.get_increment == 1

        # TODO Check if the EVM's state has changed where it should have.

        # Generate Test samples
        test_features, test_labels = toy_sim.eq_sample_n(
            test_num_each
        )

        preds = evm.predict(test_features)

        # TODO Eval Train points
        # TODO Eval Test points

        # Test EVM Saving, Loading, and Equality Comparison
        evm.save(tmp_path)
        evm_loaded = ExtremeValueMachine.load(tmp_path)
        assert evm == evm_loaded
        del evm_loaded

        # Generate the train samples
        inc_train_features, inc_train_labels = toy_sim.eq_sample_n(
            train_num_each
        )

        # Append the new train samples to the old samples
        train_features = torch.cat([train_features, inc_train_features])
        train_labels = torch.cat([train_labels, inc_train_labels])

        # Incremental fit
        evm.fit(train_features, train_labels)

        # Check increment value == 2
        assert evm.get_increment == 2

        # Generate the incremental test samples
        inc_test_features, inc_test_labels = toy_sim.eq_sample_n(test_num_each)

        # Append the new test samples to the old samples
        test_features = torch.cat([test_features, inc_test_features])
        test_labels = torch.cat([test_labels, inc_test_labels])

        preds = evm.predict(test_features)

        # TODO Eval Train points
        # TODO Eval Test points
        # TODO Eval original Train points
        # TODO Eval original Test points

        # Check save state again, confirming incremental variables are saved
        evm.save(tmp_path)
        evm_loaded = ExtremeValueMachine.load(tmp_path)
        assert evm == evm_loaded
        del evm_loaded

    #TODO def test_extreme_value_machine_with_unknowns(self):
    #    pass

    # TODO increment fit w/ unknowns. a MVN on unit cirle btwn 2 others, then
    # one opposite diagonal of this past unknown mvn, farther from unit circle
    # than any others, and one in the middle at the origin (although apparently
    # EVM does not handle zero values well!). For each, compare when fit as
    # extra_negatives and when fit with a label, so it has its own EVM.
    # Record and print out the eval scores, given deterministic, should be
    # similar scores each time.
