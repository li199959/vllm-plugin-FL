def pytest_ignore_collect(collection_path, config):
    import torch

    if (
        collection_path.name == "test_flaggems_contiguous.py"
        and not torch.cuda.is_available()
    ):
        return True
    return None
