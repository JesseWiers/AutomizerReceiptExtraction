import json
import os

"""
Loads json files
"""


def load_json(root: str, files: list, word: str) -> list:
    """
    Loads json file with given name.
    :param root: path of the files
    :param files: files to look for
    :param word: word to look for
    :return: json object
    """
    for file in files:
        if file.startswith(word):
            with open(os.path.join(root, file)) as jsonFile:
                json_object = json.load(jsonFile)
                jsonFile.close()
    return json_object
