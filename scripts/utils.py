import uuid
import json
import random
from os import listdir, walk
from os.path import isfile, join


def save_dict_to_json(dict_obj, file_path):
    with open(file_path, "w") as f:
        json.dump(dict_obj, f, indent=4)


def load_dict_from_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def generate_random_id():
    return str(uuid.uuid4())


def get_filepathes_from_dir(file_dir, include_sub_dir=False,
                            file_format=None, shuffle=False):
    
    if include_sub_dir:
        filepathes = []
        for root, _, files in walk(file_dir, topdown=False):
            for f in files:
                filepathes.append(join(root, f))
    else:
        filepathes = [join(file_dir, f) for f in listdir(file_dir)
                      if isfile(join(file_dir, f))]
        
    if file_format:
        if not isinstance(file_format, (str, list, tuple)):
            raise TypeError("file_format must be str, list or tuple.")
        file_format = tuple(file_format) if isinstance(file_format, list) else file_format
        format_checker = lambda f: f.endswith(file_format)
        filepathes = list(filter(format_checker, filepathes))

    if shuffle:
        random.shuffle(filepathes)
        
    return filepathes


def get_conversations(df, round_ix, pair_ix, 
                      return_entire_transcript=False, 
                      chunk_separator="\n\n"):
    sub = df.copy()[df["Round"] == round_ix]
    convs = sub[pair_ix].to_list()
    answer = sub["Answer"].to_list()

    if return_entire_transcript:
        return f"{chunk_separator}".join(convs), answer
    else:
        return convs, answer
