# Script to merge hdf5 chunk files to one and update info.json accordingly

import argparse
import json
import os.path
import warnings

import h5py
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="size changed")


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, help="features directory name")
parser.add_argument(
    "--skip-if-exists",
    action="store_true",
    help="whether to skip feature merging if the files exist already",
)
parser.add_argument("--chunksNum", type=int, default=16, help="number of file chunks")
parser.add_argument("--chunkSize", type=int, default=10000, help="file chunk size")
args = parser.parse_args()

if (
    args.skip_if_exists
    and os.path.exists("data/gqa_{name}.h5".format(name=args.name))
    and os.path.exists("data/gqa_{name}_merged_info.json".format(name=args.name))
    and all(
        [
            os.path.exists(
                "data/{name}/gqa_{name}_{index}.h5".format(name=args.name, index=i)
            )
            for i in range(args.chunksNum)
        ]
    )
):
    print("Using existing merged features for gqa_{}.".format(args.name))
    exit(0)

print("Merging features file for gqa_{}. This may take a while.".format(args.name))

# Format specification for features files
spec = {
    "spatial": {"features": (148855, 2048, 7, 7)},
    "objects": {"features": (148855, 100, 2048), "bboxes": (148855, 100, 4)},
}

# Merge hdf5 files
lengths = [0]
with h5py.File("data/gqa_{name}.h5".format(name=args.name), "a") as out:
    datasets = {}
    for dname in spec[args.name]:
        datasets[dname] = out.create_dataset(dname, spec[args.name][dname])

    low = 0
    for i in tqdm(range(args.chunksNum)):
        with h5py.File(
            "data/{name}/gqa_{name}_{index}.h5".format(name=args.name, index=i), "r"
        ) as chunk:
            high = low + chunk["features"].shape[0]

            for dname in spec[args.name]:
                datasets[dname][low:high] = chunk[dname][:]

            low = high
            lengths.append(high)

# Update info file
with open("data/{name}/gqa_{name}_info.json".format(name=args.name)) as infoIn:
    info = json.load(infoIn)
    for imageId in info:
        info[imageId]["index"] = lengths[info[imageId]["file"]] + info[imageId]["idx"]
        del info[imageId]["idx"]
        del info[imageId]["file"]

    with open(
        "data/gqa_{name}_merged_info.json".format(name=args.name), "w"
    ) as infoOut:
        json.dump(info, infoOut)
