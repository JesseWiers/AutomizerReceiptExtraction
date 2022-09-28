import pandas as pd
import os
from reach import Reach
import numpy as np
import json_handling

"""
Extracts data from json files and cleans the data 
The data contains annotations on images from receipts. 
"""


def word2_vec(df: pd.DataFrame) -> pd.DataFrame:
    """
    Embeds the strings found under the names column in the dataframe to 160 dimensional vectors.
    :param df: dataframe containing all the words found in all the receipts
    :return: dataframe with 160 new columns each representing one of the dimensions to which the words are embedded
    """

    strings = df['name'].to_numpy()
    vectors = []
    r = Reach.load('roularta-160.txt')

    for string in strings:
        try:
            vec = r[string]
            vectors.append(vec)
        except KeyError:
            vectors.append(np.zeros(160))

    for j in range(len(vectors[0])):  # for each 160 features
        df1 = pd.DataFrame([i[j] for i in vectors], columns=["feature{0}".format(j)])
        df = pd.concat((df1, df), axis=1)

    return df


def compute_average_coordinates(annotations: list, x: int) -> (float, float):
    """
    Computes average x,y values of. x,y values correspond with values denoting
    the corners of a square
    :param annotations: annotation file, containing the x,y coordinates
    :param x: position in the annotation file the look for
    :return: average x and average y value
    """
    avg_x = avg_y = 0

    for i in range(4):
        try:
            avg_x += annotations[x]['bounding_poly']['vertices'][i]['x'] * 0.25
            avg_y += annotations[x]['bounding_poly']['vertices'][i]['y'] * 0.25
        except KeyError:
            pass
    return avg_x, avg_y


def search_prior(words: list, idx: int) -> str:
    """
    Given a list of words, this function looks for words before a given index
    and returns the first string it found
    :param words: list of words to look into
    :param idx: index from where to start looking back
    :return: found word
    """
    i = 1
    try:
        while True:
            word = ''.join(filter(str.isalnum, words[idx - i]))
            if word.isalpha():
                return word.lower()
            i = i + 1
    except:
        print("the given data was not a receipt")
        exit()


def collect_labels(text_annotations: list, value: str) -> list:
    """
    Given a list of annotations look for the given value. If the given value is found
    the function will look for the word that comes before it.
    :param text_annotations: list annotations containing all the words in the receipt
    :param value: string to look for in the list of words
    :return: return a list containing the words that come before the given values
    """
    text = text_annotations[0]['description']  # all words
    targets = []
    words = text.split()

    for idx, word in enumerate(words):
        stripped_word = ''.join(e for e in word if e.isalnum())
        stripped_word = stripped_word.lstrip("0")

        if stripped_word == str(value):
            targets.append(search_prior(words, idx))
        else:
            try:
                if (stripped_word + words[idx + 1]) == str(value):
                    targets.append(search_prior(words, idx))
            except IndexError:
                pass

    return targets


def add_labels(text_annotations: list, value: str) -> list:
    """
    Computes a list of labels. Given a list of text annotations of a receipt and a value, computes
    which words come before the given value (total amount of receipt), if the word comes before the given value
    it gets label 1 otherwise label 0.

    :param text_annotations: list of annotations of the receipt
    :param value: string to look for
    :return: list of labels for each word
    """
    labels = []

    targets = collect_labels(text_annotations, value)

    for x in range(1, len(text_annotations)):
        if text_annotations[x]['description'].isalpha():
            if text_annotations[x]['description'].lower() in targets:
                labels.append(1)
            else:
                labels.append(0)
    return labels


def filter_data(index: int, files: list, root: str) -> pd.DataFrame:
    """
    Extracts the relevant data from annotation files and adds the data to a pandas dataframe.
    It extracts each word, the average x,y values of each word and if the path directs to the
    training labels are also extracted. Labels indicate whether after the specific word the total amount
    is displayed in the receipt.

    :param index: index of the receipt
    :param files: list containing annotations and the image of the receipt
    :param root: root of the directory
    :return: returns a dataframe, containing data on all the words in the receipt
    """

    json_handling.load_json(root, files, "annotations")
    annotation_file = json_handling.load_json(root, files, "annotations")
    vision_file = json_handling.load_json(root, files, "vision")
    text_annotations = vision_file['text_annotations']
    file_path = os.path.join(root, files[1])
    names, x_values, y_values, labels, ids, paths = [], [], [], [], [], []

    for x in range(1, len(text_annotations)):
        if text_annotations[x]['description'].isalpha():
            names.append(text_annotations[x]['description'].lower())
            x_average, y_average = compute_average_coordinates(text_annotations, x)
            x_values.append(x_average)
            y_values.append(y_average)
            ids.append(index)
            paths.append(file_path)

    d = {'id': ids, 'path': paths, 'name': names, 'x_value': x_values, 'y_value': y_values}
    df = pd.DataFrame(data=d)

    if root.startswith("data/train"):
        labels = add_labels(text_annotations, annotation_file[0]['value'])
        df['label'] = labels

    return df


def load_data(path: str) -> pd.DataFrame:
    """
    Extracts data from json files and cleans the data. Afterwards adds columns
    representing embeddings of the words found.
    param path: directory containing the json files
    :return: pandas dataframe containing the cleaned data.
    """
    pictures = []
    index = 0
    for root, dirs, files in os.walk(path):
        if len(files) > 1:
            pictures.append(filter_data(index, files, root))
            index += 1

    df = pd.concat(pictures, ignore_index=True)
    df = word2_vec(df)

    return df
