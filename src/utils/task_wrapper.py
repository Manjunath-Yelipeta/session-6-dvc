import functools
from loguru import logger

def task_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Starting {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Finished {func.__name__}")
            return result
        except Exception as e:
            logger.exception(f"Exception occurred in {func.__name__}: {str(e)}")
            raise
    return wrapper