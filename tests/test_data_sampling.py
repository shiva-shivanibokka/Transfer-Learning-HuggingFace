import pytest

torch = pytest.importorskip("torch")
# src.utils.data imports these at module load; skip cleanly if absent (e.g. CI).
pytest.importorskip("torchvision")
pytest.importorskip("datasets")


def test_stratified_subset_keeps_fraction_per_class():
    from torch.utils.data import Dataset

    from src.utils.data import _stratified_subset

    class Toy(Dataset):
        def __init__(self):
            self.labels = [i % 5 for i in range(100)]  # 20 each

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return torch.zeros(1), self.labels[i]

    sub = _stratified_subset(Toy(), fraction=0.5, seed=0)
    got = {}
    for i in range(len(sub)):
        _, lbl = sub[i]
        got[lbl] = got.get(lbl, 0) + 1
    assert all(c == 10 for c in got.values())  # 50% of 20 per class
    assert len(got) == 5
