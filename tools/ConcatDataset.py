import torch
import bisect

class ConcatDataset(torch.utils.data.dataset.Dataset):
    """
    Used to concatenate multiple datasets into one.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated
                             The first dataset -> Knowns
                             The second dataset -> Background Known Unknowns (like textures) Assigned = -1 / 11 if BG = True
                             The third dataset -> Known Unknowns (like another dataset) Assigned = -2
    """

    """
    If you get the following error message:
        AttributeError: 'int' object has no attribute 'numel'
    It is because your pytorch version needs
        torch.tensor(-1) in place of -1
        or vice versa
    """
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        all_sizes=[]
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
            print (e)
            print (len(e),s)
            print (r[-1])
            print ("###")
            all_sizes.append(len(e))
        return r, all_sizes

    def __init__(self, datasets, BG=False):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes, self.all_sizes = self.cumsum(self.datasets)
        self.BG = BG

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
            to_return=self.datasets[dataset_idx][sample_idx]
        elif dataset_idx == 1:
            if self.BG:
                sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
                to_return = (self.datasets[dataset_idx][sample_idx][0],10)
            else:
                sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
                to_return = (self.datasets[dataset_idx][sample_idx][0],-1)
        elif dataset_idx == 2:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            to_return = (self.datasets[dataset_idx][sample_idx][0],torch.tensor(-2))
        return to_return
