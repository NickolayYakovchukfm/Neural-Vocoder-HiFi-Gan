import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    batch_size = len(dataset_items)
    max_audio_length = dataset_items[0]["audio"].shape[1]
    max_mel_length = dataset_items[0]["spectrogram"].shape[1]
    result_batch = {
        "audio": torch.zeros((batch_size, max_audio_length)),
        "spectrogram": torch.zeros((batch_size, 80, max_mel_length)),
        "audio_path": [""] * batch_size,
    }

    for i in range(batch_size):
        audio = dataset_items[i]["audio"].squeeze()
        result_batch["audio"][i, : len(audio)] = audio
        result_batch["spectrogram"][i] = dataset_items[i]["spectrogram"]
        result_batch["audio_path"][i] = dataset_items[i]["audio_path"]

    return result_batch
