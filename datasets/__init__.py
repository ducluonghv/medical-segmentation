from .isic    import ISICDataset
from .glas    import GlaSDataset
from .covid   import COVIDDataset
from .lung    import LungDataset
from .dsb2018 import DSB2018Dataset

DATASET_REGISTRY = {
    'isic':    ISICDataset,
    'glas':    GlaSDataset,
    'covid':   COVIDDataset,
    'lung':    LungDataset,
    'dsb2018': DSB2018Dataset,
}

# Default configs consistent with TransAttUnet paper
DATASET_DEFAULTS = {
    'isic':    {'img_size': 256, 'n_channels': 3},
    'glas':    {'img_size': 128, 'n_channels': 3},
    'covid':   {'img_size': 512, 'n_channels': 3},
    'lung':    {'img_size': 256, 'n_channels': 3},
    'dsb2018': {'img_size': 256, 'n_channels': 3},
}
