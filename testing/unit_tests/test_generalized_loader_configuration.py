from ivadomed.loader.generalized_loader_configuration import GeneralizedLoaderConfiguration
from loguru import logger


def test_generalized_loading_configuration():
    model_dict = {
        "name": "Unet",
        "dropout_rate": 0.3,
        "bn_momentum": 0.1,
        "final_activation": "sigmoid",
        "depth": 3,
    }

    # Build a GeneralizedLoaderConfiguration:
    config: GeneralizedLoaderConfiguration = GeneralizedLoaderConfiguration(
        model_params=model_dict,
    )
    assert config.model_params == model_dict
    logger.info(config)
