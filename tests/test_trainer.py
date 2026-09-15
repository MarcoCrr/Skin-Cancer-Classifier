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
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    train_loss, train_acc = train_one_epoch(model, loader, optimizer, criterion, "cpu", scaler=scaler, use_amp=False)

    assert isinstance(train_loss, float)
    assert isinstance(train_acc, float)
    assert 0.0 <= train_acc <= 1.0


def test_train_one_epoch_updates_model():
    model = DummyModel()
    loader = get_dummy_loader()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    before = model.fc.weight.clone()

    train_one_epoch(model, loader, optimizer, criterion, "cpu", scaler=scaler, use_amp=False)

    after = model.fc.weight

    # weights should change
    assert not torch.equal(before, after)


def test_evaluate_returns_valid_accuracy():
    model = DummyModel()
    loader = get_dummy_loader()

    val_loss, val_acc = evaluate(model, loader, criterion=torch.nn.CrossEntropyLoss(), device="cpu", use_amp=False)

    assert isinstance(val_loss, float)
    assert isinstance(val_acc, float)
    assert 0.0 <= val_acc <= 1.0


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

    val_loss, val_acc = evaluate(model, loader, criterion=torch.nn.CrossEntropyLoss(), device="cpu", use_amp=False)

    assert val_acc == 1.0


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