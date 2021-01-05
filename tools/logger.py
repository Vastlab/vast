import functools
import time
import logging
import sys
from termcolor import colored
import pathlib
import atexit

def time_recorder(func):
    @functools.wraps(func)
    def time_logger(*args, **kwargs):
        start = time.time()
        logger = get_logger()
        to_return = func(*args, **kwargs)
        logger.info(f"Time taken to complete function {func.__name__} "
                    f"{time.strftime('%H:%M:%S',time.gmtime(time.time() - start))}")
        return to_return
    return time_logger


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.CRITICAL:
            return log[:(-1*len(record.message))] + " " + colored(record.message, "red", attrs=["underline"])
        return log


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def get_logger(*args, **kwargs):
    if 'distributed_rank' not in kwargs or kwargs['distributed_rank'] is None:
        return logging.getLogger()
    else:
        return setup_logger(*args, **kwargs)



@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(output=None,
                 distributed_rank=None, world_size=None,
                 color=True, level='DEBUG'):
    logger = logging.getLogger()
    if type(level)==int:
        level = logging.getLevelName(level*10)
    logger.setLevel(logging.__dict__[level])
    logger.propagate = False

    plain_formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                                        datefmt="%m/%d %H:%M:%S")
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.__dict__[level])
    if color:
        log_string = f"[%(asctime)s %(levelname)s]:{colored('%(filename)15s', 'blue')}:" \
                     f"%(lineno)3s {colored('%(funcName)20s','grey', attrs=['bold'])}"
        if distributed_rank is not None:
            log_string+=f"{colored(f'({distributed_rank}/{world_size})', 'magenta', attrs=['bold'])}"
        log_string += " -- "

        formatter = _ColorfulFormatter(log_string + "%(message)s", datefmt="%H:%M:%S")
    else:
        formatter = plain_formatter
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = pathlib.Path(output)
        else:
            filename = pathlib.Path(f"{output}/log.txt")
        if distributed_rank is not None:
            filename = pathlib.Path(f"{str(filename)}.rank{distributed_rank}")
        filename.parent.mkdir(parents=True, exist_ok=True)

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger

# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    io = filename.open(mode="a")
    atexit.register(io.close)
    return io