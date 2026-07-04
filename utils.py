import gc
import time
from contextlib import contextmanager


@contextmanager
def timer(label: str):
    # Context manager that prints how long a block took
    start = time.time()
    yield
    print(f"[{label}] {time.time() - start:.2f}s")


def downcast_dtypes(df):
    # Downcast float64 -> float32 and int64 -> int32 in place-ish (returns df).
    # Cuts memory footprint roughly in half on a wide dataset like IEEE-CIS,
    # with no meaningful precision loss for tree-based models.
    float_cols = df.select_dtypes(include="float64").columns
    int_cols = df.select_dtypes(include="int64").columns
    for col in float_cols:
        df[col] = df[col].astype("float32")
    for col in int_cols:
        df[col] = df[col].astype("int32")
    return df


def collect():
    # Force garbage collection - call after large `del`s of DataFrames
    gc.collect()
