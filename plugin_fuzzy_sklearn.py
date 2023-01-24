import os
from vacore import VACore

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymorphy2 import MorphAnalyzer

modname = os.path.basename(__file__)[:-3]  # calculating modname
morph = MorphAnalyzer()
commands = []
commands_vectors = None

# Create a Tf-Idf vectorizer
vectorizer = TfidfVectorizer()


# функция на старте
def start(core: VACore):
    manifest = {
        "name": "Fuzzy input processing with sklearn",
        "version": "1.0",
        "require_online": False,

        "fuzzy_processor": {
            "sklearn_fuzzy": (init, predict)  # первая функция инициализации, вторая - обработка
        }
    }
    return manifest


def init(core: VACore):
    global morph, commands, commands_vectors, vectorizer
    morph_commands = []

    # preprocessing step
    for keyall in core.commands.keys():
        for key in keyall.split("|"):
            morph_key = " ".join(morph.parse(word)[0].normal_form for word in key.split())
            commands.append(key)
            morph_commands.append(morph_key)

    # Vectorize the commands
    commands_vectors = vectorizer.fit_transform(morph_commands)


# The current implementation of the API requires the return of a command key, not a specific command.
# This function looks for the command key in the context and returns it.
def get_command_key_from_context(predicted_command, context):
    for keyall in context.keys():
        for key in keyall.split("|"):
            if key == predicted_command:
                return keyall
    return None


def predict(core: VACore, command: str, context: dict):
    global morph, commands, commands_vectors, vectorizer

    last_step = 0
    best_score = 0
    best_predicted_command = None
    command_by_words = command.split()+[""]

    # The first iteration of the loop predicts the full command.
    # In the next iteration of the loop, each time the last word is deleted
    # and the predicted command is checked to be unchanged.
    # This allows to extract the rest of the phrase from the command.
    for step in range(len(command_by_words), 1, -1):
        # preprocessing step
        command_to_predict = " ".join(morph.parse(word)[0].normal_form for word in command_by_words[0:step-1])

        # Vectorize the examples and the string to predict
        command_to_predict_vector = vectorizer.transform([command_to_predict])

        # Calculate the cosine similarity between the string to predict and each example
        similarities = cosine_similarity(command_to_predict_vector, commands_vectors)

        # Get the index of the most similar example
        most_similar_index = similarities.argmax()

        # Get the probability of the most similar example
        most_similar_prob = similarities.max()

        predicted_command = commands[most_similar_index]
        if best_predicted_command is None or predicted_command == best_predicted_command:
            if best_predicted_command is None:
                best_predicted_command = predicted_command
            if best_score < most_similar_prob:
                best_score = most_similar_prob
            last_step = step
        else:
            break

    # Print the most similar example and its probability
    end_of_phrase = " ".join(command_by_words[last_step-1:len(command_by_words)])
    command_key = get_command_key_from_context(best_predicted_command, context)
    best_ret = (command_key, best_score, end_of_phrase)
    return best_ret
