import functools
import time

def time_recorder(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        to_return = func(*args, **kwargs)
        print(f"Time taken to complete the function {func.__name__} : "
              f"{time.strftime('%H Hours %M Minutes %S seconds',time.gmtime(time.time() - start))}")
        return to_return
    return wrapper

