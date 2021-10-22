"""Quick interpreter testing of Extreme Value Machine."""
import torch
from vast.opensetAlgos.extreme_value_machine import ExtremeValueMachine


class ToyClassify2D4MVNs(object):
    """Create the generative sampling procedure for obtaining coordinates of
    points as feature data along with the label of which Gaussian distribution
    they belong to. The Gaussian distributions are labeled their index which
    starts at zero at the top most Gaussian centered at [0, 1] and labels the
    rest that follow clockwise around the unit circle.
    """

    def __init__(self, locs=None, scales=0.2, labels=None, seed=None):
        # TODO create PyTorch Gaussian Distributions at locs and scales
        if locs is None:
            locs = [[1, 0], [0, 1], [-1, 0], [0, -1]]

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
                torch.cat([mvn.sample_n(num) for mvn in self.mvns])[idx],
                torch.Tensor([[i] * num for i in range(len(self.mvns))]).flatten()[idx],
            )
        return (
            torch.cat([mvn.sample_n(num) for mvn in self.mvns]),
            torch.Tensor([[i] * num for i in range(len(self.mvns))]).flatten(),
        )


def setup(seed=0, device="cuda:0"):
    """The setup for all tests in this class."""
    # Set seed: Seems I cannot carry an individual RNG easily...
    torch.manual_seed(seed)

    # Create toy simulation
    toy_sim = ToyClassify2D4MVNs(seed=seed)

    # Create ExtremeValueMachine
    evm = ExtremeValueMachine(
        10,
        0.5,
        1.0,
        [0, 1, 2, 3],
        "cosine",
        device=device,
        tail_size_is_ratio=False,
    )

    return toy_sim, evm


tmp_path = "/tmp/vast/extreme_value_machine/evm.h5"
train_num_each = 10
test_num_each = 30
inc_train_num_each = 100
inc_test_num_each = 100
seed = 0
device = "cuda:0"

toy_sim, evm = setup()

train_features, train_labels = toy_sim.eq_sample_n(train_num_each)

evm.fit(train_features, train_labels)

# Generate Test samples
test_features, test_labels = toy_sim.eq_sample_n(test_num_each)

# preds = evm.predict(test_features)

kpreds = evm.known_probs(test_features)
print((kpreds.argmax(1) == test_labels).sum().tolist() / len(test_labels))

# Generate the train samples
inc_train_features, inc_train_labels = toy_sim.eq_sample_n(inc_train_num_each)

# Append the new train samples to the old samples
train_features = torch.cat([train_features, inc_train_features])
train_labels = torch.cat([train_labels, inc_train_labels])

# Incremental fit by keeping prior points
evm.fit(train_features, train_labels)

# Check increment value == 2
assert evm.get_increment == 2

# Generate the incremental test samples
inc_test_features, inc_test_labels = toy_sim.eq_sample_n(
    inc_test_num_each,
)

# Append the new test samples to the old samples
test_features = torch.cat([test_features, inc_test_features])
test_labels = torch.cat([test_labels, inc_test_labels])

# preds = evm.predict(test_features)

kpreds = evm.known_probs(test_features)
print((kpreds.argmax(1) == test_labels).sum().tolist() / len(test_labels))
