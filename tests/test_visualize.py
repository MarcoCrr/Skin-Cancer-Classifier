from src.visualize import load_config, load_training_log


def test_load_config(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("system:\n  device: cpu\n")

    config = load_config(config_file)

    assert config["system"]["device"] == "cpu"


def test_load_training_log(tmp_path):
    log_file = tmp_path / "log.txt"
    log_file.write_text("0,0.5,0.6\n1,0.4,0.7\n")

    epochs, losses, accs = load_training_log(log_file)

    assert epochs == [0, 1]
    assert losses == [0.5, 0.4]
    assert accs == [0.6, 0.7]