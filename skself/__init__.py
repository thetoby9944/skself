__version__ = "0.1.0"


if __name__ == '__main__':
    pass


def enable_hyper_search_progress_bar():
    """
    Progress bar for sklearn parallel searches in jupyter notebooks.

    This fixes two issues:

    1. Notebooks unrealiably displas output from subprocesses
    2. Parallel Processing does not have a process bar in sklearn

    WARNING. This sets the default parallelization in sklearn to threading.
    That's okay unless you want to utilize the processing power of multiple machines.
    """
    from sklearn.utils import parallel_backend
    import sklearn.model_selection as ms
    from tqdm import tqdm

    class ParallelProgressBar(ms._search.Parallel):
        def __call__(self, it):
            return super().__call__(tqdm(list(it)))

    # Set backend to threading to get subprocess outputs in jupyter
    parallel_backend("threading")

    # Monkey patch parallel call so we can see our progress bar
    if not isinstance(ms._search.Parallel, ParallelProgressBar):
        ms._search.Parallel = ParallelProgressBar
