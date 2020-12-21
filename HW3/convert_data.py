import json
from typing import List, Union, Tuple, NoReturn

JSON_PATH = 'data/qa_data.jsonl'
SOURCE_PATH = 'data/source'
TARGET_PATH = 'data/target'


def process_question(q: dict) -> Union[Tuple[List[str], List[str]], None]:

    if 'responses' not in q.keys():
        return None
    else:
        question = q['question']
        responses = q['responses']
        sources = []
        targets = []
        for element in responses:
            sources.append(question)
            targets.append(element)

        sources = [element + '\n' for element in sources]
        targets = [element + '\n' for element in targets]

        return sources, targets


def empty_file(path):
    open(path, 'w').close()


def write_question(sources: List[str],
                   targets: List[str],
                   path_to_source: str,
                   path_to_target: str) -> NoReturn:

    with open(path_to_source, 'a') as file:
        file.writelines(sources)

    with open(path_to_target, 'a') as file:
        file.writelines(targets)


def convert_json(path_to_read: str,
                 path_to_source: str,
                 path_to_target: str,
                 max_lines: int) -> NoReturn:

    # create empty files to write to
    empty_file(path_to_source)
    empty_file(path_to_target)

    with open(path_to_read) as file:
        for index, line in enumerate(file):
            if index >= max_lines:
                break
            if line:
                sources, targets = process_question(json.loads(line))
                write_question(sources, targets, path_to_source, path_to_target)


convert_json(JSON_PATH, SOURCE_PATH, TARGET_PATH, 150_000)

