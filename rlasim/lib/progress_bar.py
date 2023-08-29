from typing import Callable
import time
import threading
import math


class AsyncProgressBar:
    def __init__(self, total, progress_fn : Callable, prefix:str= ''):
        self.total = total
        self.len_fn = progress_fn
        self.prefix = prefix
    def format_time(self, seconds):
        # m, s = divmod(seconds, 60)
        # return f"{int(m):02d}:{int(s):02d}"
        return "%.2f"%seconds

    def update_progress(self):
        self.length = 40
        progress = float(self.len_fn())
        # if progress == 0:
        #     return
        fill = '█'
        elapsed_time = time.time() - self.start_time
        if progress != 0:
            remaining_time = (elapsed_time / progress) * (self.total - progress)
            progress_bar = fill * int(progress / self.total * self.length)
        else:
            remaining_time = math.inf
            progress_bar = ''

        spaces = ' ' * (self.length - len(progress_bar))
        print(f"\r{self.prefix}[{progress_bar}{spaces}] {progress}/{self.total} "
              f"({progress / self.total:.1%}) Elapsed: {self.format_time(elapsed_time)} "
              f"Remaining: {self.format_time(remaining_time)}", end='', flush=True)

    def thread_fn(self):
        while True:
            if self.kill_signal:
                self.update_progress()
                break
            self.update_progress()
            time.sleep(0.05)

    def __enter__(self):
        self.start_time = time.time()
        self.kill_signal = False
        self.my_thread = threading.Thread(target=self.thread_fn)
        self.my_thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # self.end_time = time.time()
        # elapsed_time = self.end_time - self.start_time
        # print(f"Time taken: {elapsed_time:.4f} seconds")
        self.kill_signal = True
        self.my_thread.join()