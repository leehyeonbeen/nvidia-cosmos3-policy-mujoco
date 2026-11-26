"""Download DROID failure/success retry pairs: a failure followed, on the same robot and scene, by a success of the same
task within 2 minutes. The failure gets the success's crowd annotation as its prompt (DROID annotates successes only).
Runs in the policy server's env (h5py).

    python scripts/download_droid_pairs.py --pairs 2 --output-dir outputs/_test/droid_pairs

Writes <output-dir>/{failure,success}/<lab>_<label>_<date>_<folder>/ (metadata, trajectory.h5, the 3 camera MP4s) and
<output-dir>/pairs.json: failure folder, success folder and prompt per pair. Reads DROID's crowd annotations from
datasets/droid_sample/aggregated-annotations-030724.json.
"""

import argparse
import json
import os
import random
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

BUCKET = "https://storage.googleapis.com/storage/v1/b/gresearch/o"
FILES = "https://storage.googleapis.com/gresearch/"
RAW = "robotics/droid_raw/1.0.1/"
ANNOTATIONS = "datasets/droid_sample/aggregated-annotations-030724.json"
MAX_GAP = 120  # seconds from the failure's start to the success's start
# Failures that moved at most 0.1 m were stopped before an attempt; the least a success moved here was 0.65 m.
MIN_TRAVEL = 0.2
MIN_FAILURE_SECONDS = 10  # about 5 Policy-DROID calls (32 actions at 15 fps)


def list_metadata(lab, label):
    """Lists the names of a lab's metadata files for one label.

    Args:
        lab: The lab's folder in DROID's raw data.
        label: "failure" or "success".

    Returns:
        The metadata files' names in the bucket.
    """
    names, token = [], None
    while True:
        query = dict(
            prefix=f"{RAW}{lab}/{label}/",
            matchGlob="**/metadata*.json",
            fields="items(name),nextPageToken",
            maxResults=5000,
        )
        if token:
            query["pageToken"] = token
        page = json.load(
            urllib.request.urlopen(f"{BUCKET}?{urllib.parse.urlencode(query)}")
        )
        names += [item["name"] for item in page.get("items", [])]
        token = page.get("nextPageToken")
        if not token:
            return names


def robot_and_time(name):
    """Returns (lab, user id) and start time, from metadata_<lab>+<user id>+<date>-<h>h-<m>m-<s>s.json.

    Args:
        name: A metadata file's name.

    Returns:
        (lab, user id) and the start time (datetime), or (None, None) if the name does not match.
    """
    match = re.search(
        r"metadata_(.+?)\+(\w+)\+(\d{4}-\d{2}-\d{2})-(\d+)h-(\d+)m-(\d+)s\.json", name
    )
    if not match:
        return None, None
    lab, user, date, h, m, s = match.groups()
    return (lab, user), datetime.fromisoformat(f"{date} {int(h):02d}:{m}:{s}")


def candidates(lab, annotations):
    """Pairs each annotated success with the failure right before it on the same robot.

    A pair is kept only if that failure started within MAX_GAP and no other episode came in between.

    Args:
        lab: The lab's folder in DROID's raw data.
        annotations: DROID's crowd annotations, by episode uuid.

    Returns:
        A list of (failure, success) metadata names.
    """
    episodes = []  # (robot, start time, metadata name, label), sorted by robot, then time
    for label in ("failure", "success"):
        for name in list_metadata(lab, label):
            robot, start = robot_and_time(name)
            if robot:
                episodes.append((robot, start, name, label))
    episodes.sort()
    pairs = []
    for i in range(1, len(episodes)):
        robot_before, start_before, failure, label_before = episodes[i - 1]
        robot_after, start_after, success, label_after = episodes[i]
        uuid = re.search(r"metadata_(.+)\.json", success).group(1)
        if (
            robot_before == robot_after
            and label_before == "failure"
            and label_after == "success"
            and uuid in annotations
            and (start_after - start_before).total_seconds() <= MAX_GAP
        ):
            pairs.append((failure, success))
    return pairs


def fetch_json(name):
    return json.load(urllib.request.urlopen(FILES + urllib.parse.quote(name)))


def local_folder(output_dir, metadata_name):
    """Returns the episode's local folder, <lab>_<label>_<date>_<folder>, as in datasets/droid_sample.

    Args:
        output_dir: The folder the pairs are saved in.
        metadata_name: The episode's metadata name in the bucket.

    Returns:
        <output_dir>/<label>/<lab>_<label>_<date>_<folder>, with ":" in <folder> written as "-".
    """
    lab, label, date, folder = metadata_name.removeprefix(RAW).split("/")[:4]
    return f"{output_dir}/{label}/{lab}_{label}_{date}_{folder.replace(':', '-')}"


def episode_files(metadata_name, meta):
    episode = metadata_name.rsplit("/", 1)[0] + "/"
    return [metadata_name, episode + "trajectory.h5"] + [
        episode + "recordings/MP4/" + os.path.basename(meta[f"{c}_mp4_path"])
        for c in ("wrist", "ext1", "ext2")
    ]


def gripper_travel(metadata_name):
    """Returns the summed per-axis gripper motion (m).

    Near zero: a recording stopped before any attempt.

    Args:
        metadata_name: The episode's metadata name in the bucket; its trajectory.h5 is read.

    Returns:
        The summed motion (m), a float.
    """
    import io

    import h5py
    import numpy as np

    url = FILES + urllib.parse.quote(metadata_name.rsplit("/", 1)[0] + "/trajectory.h5")
    with h5py.File(io.BytesIO(urllib.request.urlopen(url).read())) as h5:
        xyz = h5["observation/robot_state/cartesian_position"][:, :3]
    return float(np.abs(np.diff(xyz, axis=0)).sum())


def complete(metadata_name, meta):
    """Checks whether the bucket has all the episode's files.

    DROID's raw copy misses some (https://droid-dataset.github.io/droid/the-droid-dataset).

    Args:
        metadata_name: The episode's metadata name in the bucket.
        meta: The episode's metadata, which names its camera MP4s.

    Returns:
        True if the bucket has the metadata, trajectory.h5 and the 3 camera MP4s.
    """
    query = dict(prefix=metadata_name.rsplit("/", 1)[0] + "/", fields="items(name)")
    page = json.load(
        urllib.request.urlopen(f"{BUCKET}?{urllib.parse.urlencode(query)}")
    )
    return set(episode_files(metadata_name, meta)) <= {
        item["name"] for item in page.get("items", [])
    }


def download_episode(output_dir, metadata_name, meta):
    folder = local_folder(output_dir, metadata_name)
    os.makedirs(folder, exist_ok=True)
    for name in episode_files(metadata_name, meta):
        path = f"{folder}/{os.path.basename(name)}"
        if not os.path.exists(path):
            urllib.request.urlretrieve(FILES + urllib.parse.quote(name), path + ".part")
            os.rename(path + ".part", path)
    return os.path.basename(folder)


def main(pairs, output_dir, seed):
    annotations = json.load(open(ANNOTATIONS))
    labs = [
        p.removeprefix(RAW).rstrip("/")
        for p in json.load(
            urllib.request.urlopen(f"{BUCKET}?prefix={RAW}&delimiter=/")
        )["prefixes"]
    ]
    rng = random.Random(seed)
    queues = {}
    for lab in labs:
        queues[lab] = candidates(lab, annotations)
        rng.shuffle(queues[lab])
        print(f"{lab}: {len(queues[lab])} candidates", flush=True)

    def download(failure, f, success, s):
        _, failure_start = robot_and_time(failure)
        _, success_start = robot_and_time(success)
        row = dict(
            failure=download_episode(output_dir, failure, f),
            success=download_episode(output_dir, success, s),
            prompt=annotations[s["uuid"]]["language_instruction1"],
            current_task=s["current_task"],
            scene_id=s["scene_id"],
            seconds_apart=(success_start - failure_start).total_seconds(),
        )
        print(f"downloaded {row['failure']} + {row['success']}", flush=True)
        return row

    # round robin over the labs, so each lab gives about the same number of pairs
    jobs = []
    with ThreadPoolExecutor(8) as pool:
        while len(jobs) < pairs and any(queues.values()):
            for lab in labs:
                if len(jobs) == pairs or not queues[lab]:
                    continue
                failure, success = queues[lab].pop()
                f, s = fetch_json(failure), fetch_json(success)
                if (
                    f["current_task"] == s["current_task"]
                    and f["scene_id"] == s["scene_id"]
                    and f["trajectory_length"] / 15 >= MIN_FAILURE_SECONDS
                    and annotations[s["uuid"]]["language_instruction1"]
                    .strip(" .")
                    .lower()
                    != "no action"
                    and complete(failure, f)
                    and complete(success, s)
                    and gripper_travel(failure) >= MIN_TRAVEL
                ):
                    jobs.append(pool.submit(download, failure, f, success, s))
                    print(f"pair {len(jobs)}/{pairs}: {lab}", flush=True)
        table = [job.result() for job in jobs]
    with open(f"{output_dir}/pairs.json", "w") as f:
        json.dump(table, f, indent=1)
    print(f"{output_dir}/pairs.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--output-dir", default="datasets/droid_sample", help="where the pairs are saved")
    parser.add_argument("--seed", type=int, default=0)
    main(**vars(parser.parse_args()))
