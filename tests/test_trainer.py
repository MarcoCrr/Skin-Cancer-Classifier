import torch

from src.trainer import train_one_epoch, evaluate, should_stop_early


# -------------------------
# Dummy components
# -------------------------

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


def get_dummy_loader(batch_size=4, num_samples=20):
    x = torch.randn(num_samples, 10)
    y = torch.randint(0, 2, (num_samples,))
    dataset = list(zip(x, y))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


# -------------------------
# Tests
# -------------------------

def test_train_one_epoch_runs():
    model = DummyModel()
    loader = get_dummy_loader()

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    loss = train_one_epoch(model, loader, optimizer, criterion, "cpu")

    assert isinstance(loss, float)
    assert loss > 0


def test_train_one_epoch_updates_model():
    model = DummyModel()
    loader = get_dummy_loader()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    before = model.fc.weight.clone()

    train_one_epoch(model, loader, optimizer, criterion, "cpu")

    after = model.fc.weight

    # weights should change
    assert not torch.equal(before, after)


def test_evaluate_returns_valid_accuracy():
    model = DummyModel()
    loader = get_dummy_loader()

    acc = evaluate(model, loader, "cpu")

    assert 0.0 <= acc <= 1.0


def test_evaluate_perfect_accuracy():
    class PerfectModel(torch.nn.Module):
        def forward(self, x):
            # Always predicts class 0
            return torch.tensor([[10.0, 0.0]] * x.shape[0])

    model = PerfectModel()

    # Create dataset where all labels = 0
    x = torch.randn(10, 10)
    y = torch.zeros(10, dtype=torch.long)
    loader = torch.utils.data.DataLoader(list(zip(x, y)), batch_size=2)

    acc = evaluate(model, loader, "cpu")

    assert acc == 1.0


def test_should_stop_early_improvement():
    improved, counter = should_stop_early(val_acc=0.8, best_acc=0.5, counter=2)

    assert improved is True
    assert counter == 0


def test_should_stop_early_no_improvement():
    improved, counter = should_stop_early(val_acc=0.5, best_acc=0.8, counter=2)

    assert improved is False
    assert counter == 3


def test_should_stop_early_equal_accuracy():
    improved, counter = should_stop_early(val_acc=0.5, best_acc=0.5, counter=1)

    assert improved is False
    assert counter == 2