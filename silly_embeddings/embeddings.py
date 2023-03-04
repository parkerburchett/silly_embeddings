import pandas as pd
import numpy as np
import threading
import queue
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import time
import dotenv

dotenv.load_dotenv()

import os

if not torch.cuda.is_available():
    raise ValueError(f"{torch.cuda.is_available()=}")


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    raise ValueError("on cpu")
CACHE_FOLDER = os.environ['CACHE_FOLDER'] # "/home/parker/Documents/GitHub/small_embeddings/models"


class Worker:
    def __init__(
        self, embedding_col: str, in_queue: queue.Queue, out_queue: queue.Queue
    ):
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            cache_folder=CACHE_FOLDER,
            device=DEVICE,
        )
        self.embedding_col = embedding_col
        self.in_queue = in_queue
        self.out_queue = out_queue

    def run(self) -> None:
        """Process data in the input queue and put it into the output queue with the features embedded"""
        while self.in_queue.not_empty:
            try:
                df_to_encode = self.in_queue.get_nowait()
            except queue.Empty:
                break
            embedding = self.model.encode(df_to_encode[self.embedding_col].values)
            embedded_df = pd.DataFrame(embedding)
            embedded_df.columns = [f"feature_{i}" for i in embedded_df.columns]
            embedded_df.index = df_to_encode.index
            df_with_embedding = pd.concat([df_to_encode, embedded_df], axis=1)
            df_with_embedding["embedding_col"] = self.embedding_col
            self.out_queue.put_nowait(df_with_embedding)
            self.in_queue.task_done()


class Consumer:
    def __init__(self, out_queue: queue.Queue, num_groups: int) -> None:
        self.out_queue = out_queue
        self.num_groups = num_groups
        self.processed_df: pd.DataFrame = None

    def run(self) -> None:
        count = 0
        with tqdm(total=self.num_groups) as pbar:
            while count < self.num_groups:
                try:
                    df_with_embedding = self.out_queue.get_nowait()
                    count += 1
                    if self.processed_df is None:
                        self.processed_df = df_with_embedding
                    else:
                        self.processed_df = pd.concat(
                            [self.processed_df, df_with_embedding], axis=0
                        )
                    pbar.update(1)
                except queue.Empty:
                    time.sleep(5)


def _build_embedded_df(
    workers: list[Worker], output_queue: queue.Queue, num_groups: int
) -> pd.DataFrame:
    """build a dataframe where one column is col_to_embed and the other columns are the embedding colusmn"""
    threads = [threading.Thread(target=worker.run) for worker in workers]
    for thread in threads:
        thread.start()

    consumer = Consumer(output_queue, num_groups)
    consumer_thread = threading.Thread(target=consumer.run)
    consumer_thread.start()

    for thread in threads:
        thread.join()
    consumer_thread.join()
    return consumer.processed_df


def _compute_num_groups(df: pd.DataFrame, num_workers: int) -> int:
    desired_size = 10_000
    # I want at most 10k rows per group
    if num_workers * desired_size > len(df):
        return num_workers
    else:
        return len(df) // desired_size


def compute_threaded_embeddings(
    df: pd.DataFrame,
    embedding_col: str,
    num_workers: int = 8,
) -> pd.DataFrame:
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    workers = [
        Worker(embedding_col, input_queue, output_queue) for _ in range(num_workers)
    ]
    num_groups = _compute_num_groups(df, num_workers)

    to_encode_sub_dfs = np.array_split(df, num_groups)
    print(
        f"Using {num_workers=} {num_groups=} group size {to_encode_sub_dfs[0].shape=}"
    )
    for text_to_encode in to_encode_sub_dfs:
        input_queue.put(text_to_encode)

    return _build_embedded_df(workers, output_queue, num_groups)
