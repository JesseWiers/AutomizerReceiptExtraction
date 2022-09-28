import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import json_handling

"""
Extracts the predicted total amounts of the receipts in the given directory path 
"""


def get_predictions(path: str, targets: list) -> tuple:
    """
    Predicts the total amount of a receipt based on the words in the given list. The words
    tell that the total amount comes after that word in the annotation file. The function only
    looks for numbers after the word since the goal is to look for the total amount.
    :param path: path of the annotation files
    :param targets: words to look for
    :return: tuple of predicted total amounts and paths of the receipts
    """
    results, vision_files = [], []

    for root, dirs, files in os.walk(path):
        if len(files) > 1:
            vision_files.append(json_handling.load_json(root, files, "vision"))

    # Loops over all the targets for the different receipts. If the target matches with a word in the annotation file
    # the first integer that is found after the word is added to the predictions. If no integer is found after the word
    # the prediction becomes 0
    for i in range(len(targets)):
        text_annotations = vision_files[i]['text_annotations']
        text = text_annotations[0]['description']
        words = text.split()

        for idx, word in enumerate(words):
            stripped_word = ''.join(e for e in word if e.isalpha())
            stripped_word = stripped_word.lstrip("0")
            stripped_word = stripped_word.lower()

            prediction = str(targets[i][1])
            if stripped_word == prediction:
                for j in range(idx + 1, len(words)):
                    word = words[j].lstrip("€$")
                    try:
                        if word[0].isdigit():
                            word = ''.join(c for c in word if c.isdigit())
                            results.append(word)
                            break
                    except IndexError:
                        pass
                else:
                    results.append(0)
                    break
                break

    directories = os.listdir(path)

    return directories, results


def get_targets(df: pd.DataFrame, model: LogisticRegression) -> list:
    """
    Extracts the words with the highest likelihood of coming before the total amount per
    receipt and returns a list of these words with the paths.
    :param df: dataframe with all the receipts
    :param model: model to make predictions on the dataframe
    :return: list of paths and predicted words per receipt
    """
    x = df.drop(['name', 'id', 'path'], axis='columns')
    probabilities = model.predict_proba(x)

    df["odds_0"] = [i[0] for i in probabilities]
    df["odds_1"] = [i[1] for i in probabilities]

    predictions = []

    for i in range(df['id'].nunique()):
        check = df.loc[df['id'] == i]  # Get each picture (id)
        row = check['odds_1'].idxmax()  # for the picture find the row with the highest odds1
        predictions.append([df.iloc[row]['path'], df.iloc[row]['name']])
    return predictions


def predict(df: pd.DataFrame, model: LogisticRegression, path: str) -> tuple:
    """
    Evaluates a model by taking for each file the word that has the highest likelihood of being the
    word before the total amount. Then the first encountered number after this word is extracted after which
    it is added to a list. This list combined with the paths of the pictures is then returned.
    :param df: dataframe to collect the
    :param model: model to make predictions on the dataframe
    :param path: path of the data
    :return: list of predicted total amounts and paths of the receipts
    """
    target_words = get_targets(df, model)
    predictions = get_predictions(path, target_words)
    return predictions
