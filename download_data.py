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


def unpack(tarname: pathlib.Path, destination: pathlib.Path):
    # recursive function to unpack all tar.gz files in a directory
    print("unpacking ", tarname, destination)
    if tarname.suffixes != [".tar", ".gz"]:
        # stop if this is not a compressed directory
        return
    tar = tarfile.open(tarname, "r:gz")
    tar.extractall(path=destination)
    tar.close()

    # for each file in destination: call unpack again
    outdir = destination / tarname.name.replace(".tar.gz", "")

    for file in outdir.iterdir():
        unpack(file, outdir)


def move_and_unpack_data(tmpdir: pathlib.Path, src_dir: str, filename: str, unpack_data: bool):
    data_src = tmpdir / src_dir / filename
    data_dst = pathlib.Path(".")
    shutil.copy(data_src, data_dst)

    if unpack_data:
        unpack(data_dst / filename, data_dst.parent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unpack",
        help="If set, unpack all compressed subdirectories as well. This will require approx. 30 GB of disk space.",
        action="store_true"
    )

    unpack_data = parser.parse_args().unpack
    download_path = "https://cme.h-its.org/exelixis/material/simulation_study.tar.gz"

    with tempfile.TemporaryDirectory() as tmpdir:
        print("Downloading data from ", download_path)
        filename, _ = urllib.request.urlretrieve(url=download_path, reporthook=progress)

        print("\nUnpacking data")
        tar = tarfile.open(filename, "r:gz")
        tar.extractall(path=tmpdir)
        tar.close()

        tmpdir = pathlib.Path(tmpdir)
        move_and_unpack_data(tmpdir=tmpdir, src_dir="supplementary_data", filename="input_data.tar.gz", unpack_data=unpack_data)
        move_and_unpack_data(tmpdir=tmpdir, src_dir="supplementary_data/GBT", filename="dataframes.tar.gz", unpack_data=unpack_data)
        move_and_unpack_data(tmpdir=tmpdir, src_dir="supplementary_data/GBT", filename="training_results.tar.gz", unpack_data=unpack_data)
