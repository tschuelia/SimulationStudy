import argparse
import pathlib
import shutil
import urllib.request
import tarfile
import tempfile
import time

global start_time


def progress(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    print("%d%%, %d MB, %d KB/s, total time: %d seconds" % (percent, progress_size / (1024 * 1024), speed, duration), end="\r")



if __name__ == "__main__":
    download_path = "https://cme.h-its.org/exelixis/material/simulations/tb_with_gaps_sparta.tar.gz"

    parser = argparse.ArgumentParser()

    parser.add_argument("--unpack", action="store_true", help="Unpacks all ")



    with tempfile.TemporaryDirectory() as tmpdir:
        print("Downloading data from ", download_path)
        filename, _ = urllib.request.urlretrieve(url=download_path, reporthook=progress)

        print("\nUnpacking data")
        tar = tarfile.open(filename, "r:gz")
        tar.extractall(path=tmpdir)
        tar.close()

        input_data = pathlib.Path(tmpdir) / "input_data"
        shutil.move(input_data, "input_data")

        gbt_input_dataframes = pathlib.Path(tmpdir) / "GBT" / "dataframes"
        shutil.move(gbt_input_dataframes, "dataframes")

        gbt_training_results = pathlib.Path(tmpdir) / "GBT" / "training_results"
        shutil.move(gbt_training_results, "training_results")

