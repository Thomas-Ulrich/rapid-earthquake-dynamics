#!/usr/bin/env python3
import json
import os
import wget
import argparse
import shutil
import numpy as np
from obspy import UTCDateTime
from datetime import datetime, timezone


def find_key_recursive(data, target_key, current_path=None):
    if current_path is None:
        current_path = []

    occurrences = []

    if isinstance(data, dict):
        for key, value in data.items():
            new_path = current_path + [str(key)]
            if key == target_key:
                occurrences.append("|".join(new_path))
            occurrences.extend(find_key_recursive(value, target_key, new_path))
    elif isinstance(data, list):
        for index, item in enumerate(data):
            new_path = current_path + [str(index)]
            occurrences.extend(find_key_recursive(item, target_key, new_path))

    return occurrences


def get_value_by_key(data, target_key):
    keys = target_key.split("|")
    current_data = data

    for key in keys:
        if isinstance(current_data, dict) and key in current_data:
            current_data = current_data[key]
        else:
            # Key not found, return None or raise an exception based on your needs
            return None

    return current_data


def get_value_from_usgs_data(jsondata, key):
    item = find_key_recursive(jsondata, key)[0]
    return get_value_by_key(jsondata, item)


def wget_overwrite(url, out_fname=None):
    fn = out_fname if out_fname else os.path.basename(url)
    if os.path.exists(fn):
        os.remove(fn)
    wget.download(url, out=out_fname, bar=None)


def retrieve_usgs_id_from_dtgeo_dict(fname, min_mag):
    ev = np.load(fname, allow_pickle=True).item()

    origin_time = UTCDateTime(ev["ot"])
    origin_time_plus_one_day = origin_time + 24 * 3600

    url = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson"
    url += f"&minmagnitude={min_mag}"
    url += f"&starttime={origin_time}&endtime={origin_time_plus_one_day}"
    url += f"&minlatitude={ev['lat'] - 0.5}&maxlatitude={ev['lat'] + 0.5}"
    url += f"&minlongitude={ev['lon'] - 0.5}&maxlongitude={ev['lon'] + 0.5}"

    fn_json = "out.json"
    wget_overwrite(url, fn_json)

    with open(fn_json) as f:
        jsondata = json.load(f)
    features = get_value_from_usgs_data(jsondata, "features")
    if not features:

        def convert(obj):
            # Convert NumPy arrays to lists before pretty printing
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        pretty_dict = json.dumps(dict(ev), indent=4, sort_keys=True, default=convert)
        print(pretty_dict)
        raise ValueError(f"usgs event_id could not be retrieved from {fname}")
    usgs_id = features[0]["id"]
    return usgs_id


def get_data(
    usgs_id_or_dtgeo_npy,
    min_magnitude,
    suffix,
    use_usgs_finite_fault=True,
    download_usgs_fsp=False,
):
    if usgs_id_or_dtgeo_npy[-3:] == "npy":
        usgs_id = retrieve_usgs_id_from_dtgeo_dict(
            args.usgs_id_or_dtgeo_npy, min_magnitude
        )
    else:
        splited_code = usgs_id_or_dtgeo_npy.split("_")
        if len(splited_code) == 2:
            usgs_id = splited_code[0]
            finite_fault_code = usgs_id_or_dtgeo_npy
            print(f"given code {finite_fault_code} describes a usgs finite fault model")
        elif len(splited_code) == 1:
            usgs_id = splited_code[0]
            finite_fault_code = None
        else:
            raise ValueError("unexpected structure of usgs_code")

    url = (
        "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson"
        f"&&minmagnitude={min_magnitude}&eventid={usgs_id}"
    )
    fn_json = f"{usgs_id}.json"
    wget_overwrite(url, fn_json)

    with open(fn_json) as f:
        jsondata = json.load(f)

    mag = get_value_from_usgs_data(jsondata, "mag")
    place = get_value_from_usgs_data(jsondata, "place")

    # Convert to timestamp in milliseconds seconds
    eventtime = float(get_value_from_usgs_data(jsondata, "time")) / 1000.0
    day = datetime.fromtimestamp(eventtime, tz=timezone.utc).strftime("%Y-%m-%d")

    descr = "_".join(place.split(",")[-1].split())
    # remove parenthesis in descr
    descr = descr.replace("(", "").replace(")", "")

    if use_usgs_finite_fault or download_usgs_fsp:
        finite_faults = get_value_from_usgs_data(jsondata, "finite-fault")
        if finite_fault_code:
            availables = [finite_fault["code"] for finite_fault in finite_faults]
            if finite_fault_code not in availables:
                raise ValueError(f"{finite_fault_code} not found in {availables}")
            else:
                ff_id = availables.index(finite_fault_code)
        else:
            # if not specified we use the most recently updated
            update_times = [ff["updateTime"] for ff in finite_faults]
            ff_id = update_times.index(max(update_times))

        finite_fault = finite_faults[ff_id]
        code_finite_fault = finite_fault["code"]
        update_time = finite_fault["updateTime"]
        hypocenter_x = finite_fault["properties"]["longitude"]
        hypocenter_y = finite_fault["properties"]["latitude"]
        hypocenter_z = finite_fault["properties"]["depth"]
    else:
        code_finite_fault = usgs_id

    folder_name = f"{day}_Mw{mag}_{descr[:20]}_{code_finite_fault}{suffix}"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    if not os.path.exists(f"{folder_name}/tmp"):
        os.makedirs(f"{folder_name}/tmp")

    # we use the first released hypoccenter for the projection, to avoid having change
    # in the projection if there is an update
    origin = get_value_from_usgs_data(jsondata, "origin")
    # first_released_index = min(
    #    range(len(origin)), key=lambda i: origin[i]["updateTime"]
    # )
    lon = float(origin[0]["properties"]["longitude"])
    lat = float(origin[0]["properties"]["latitude"])

    projection = f"+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0={lon:.2f} +lat_0={lat:.2f}"

    if use_usgs_finite_fault:
        with open(f"{folder_name}/tmp/hypocenter.txt", "w") as f:
            jsondata = f.write(f"{hypocenter_x} {hypocenter_y} {hypocenter_z}\n")

        for fn in [
            "moment_rate.mr",
            "basic_inversion.param",
            "complete_inversion.fsp",
        ]:
            url = (
                f"https://earthquake.usgs.gov/product/finite-fault/{code_finite_fault}"
                f"/us/{update_time}/{fn}"
            )
            wget_overwrite(url, f"{folder_name}/tmp/{fn}")

    elif download_usgs_fsp:
        for fn in ["complete_inversion.fsp"]:
            url = (
                f"https://earthquake.usgs.gov/product/finite-fault/{code_finite_fault}"
                f"/us/{update_time}/{fn}"
            )
            wget_overwrite(url, f"{folder_name}/tmp/{fn}")

    shutil.move(fn_json, f"{folder_name}/tmp/{fn_json}")
    print(folder_name)
    derived_config = {}
    derived_config["folder_name"] = folder_name
    derived_config["projection"] = projection

    return derived_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="download usgs finite model data for a specific earthquake"
    )
    parser.add_argument(
        "usgs_id_or_dtgeo_npy",
        help="usgs earthquake code or event dictionnary (dtgeo workflow)",
    )
    parser.add_argument(
        "--min_magnitude",
        nargs=1,
        help="min magnitude in eq query",
        default=[7.0],
        type=float,
    )
    parser.add_argument("--suffix", nargs=1, help="suffix for folder name")
    args = parser.parse_args()
    folder = get_data(
        args.usgs_id_or_dtgeo_npy,
        args.min_magnitude[0],
        args.suffix[0] if args.suffix else "",
    )
