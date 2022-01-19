# -*-coding:utf-8-*-

"""
    process genia jsonlines format into NER json format.
    for example:
        {
            "doc_key": "ge/train/0001",
            "ners": [[[0, 1, "DNA"], [4, 5, "protein"]], [[3, 3, "protein"]]],
            "sentences": [["IL-2", "gene", "expression", "and", "NF-kappa", "B", "activation", "through", "CD28", "."],
            ["Activation", "of", "the", "CD28", "surface", "receptor", "provides", "a", "costimulatory", "signal", "."]]
         }
         ======>
         [
            {
                "doc_key": "ge/train/0001",
                "text": "IL-2 gene expression and NF-kappa B activation through CD28 .",
                "entities": [
                    {"start_idx": 0, "end_idx": 1, "type": DNA, "entity": "IL-2 gene"},
                    {"start_idx": 4, "end_idx": 5, "type": "PROTEIN", "entity": "NF-kappa"}
                ]
             },
            {
                "doc_key": "ge/train/0001",
                "text": "Activation of the CD28 surface receptor provides a costimulatory signal .",
                "entities": [
                    {"start_idx": 3, "end_idx": 3, "type": PROTEIN, "entity": "CD28"}
                ]
             },
             ...
         ]
"""
import json
import os

from tqdm import tqdm


def process_genia_file(filepath):
    ret_list = []
    with open(filepath, 'r', encoding='utf8') as fin:
        for ith, line in enumerate(tqdm(fin)):
            json_dict = json.loads(line.strip())
            doc_key = json_dict["doc_key"]
            ners = json_dict["ners"]
            sentences = json_dict["sentences"]
            assert len(ners) == len(sentences)

            for entities, sentence in zip(ners, sentences):
                entities = [{"start_idx": ent[0],
                             "end_idx": ent[1],
                             "type": ent[2].upper(),
                             "entity": ' '.join(sentence[ent[0]: ent[1] + 1])
                             }
                            for ent in entities]
                ret_dict = {
                    "doc_key": doc_key,
                    "text": ' '.join(sentence),
                    "entities": entities
                }
                ret_list.append(ret_dict)
    return ret_list


def process_genia_dataset(dir_path='.'):
    train_dev_test_dict = {"train": [], "dev": [], "test": []}
    tag_set = set()
    for file in os.listdir(dir_path):
        if file.endswith(".jsonlines"):
            filepath = os.path.join(dir_path, file)
            examples = process_genia_file(filepath)
            for ex in examples:
                tag_set |= set(ent["type"] for ent in ex['entities'])
                for key in train_dev_test_dict.keys():
                    if key in ex['doc_key']:
                        train_dev_test_dict[key].append(ex)

    for key, value in train_dev_test_dict.items():
        with open(f"{key}.json", "w", encoding='utf8') as fout:
            json.dump(value, fout, indent=2, ensure_ascii=False)
            print(f"write {key}.json done: {len(value)} examples")
    with open('tag.txt', 'w', encoding='utf8') as fout:
        tag_list = sorted(list(tag_set))
        fout.write('\n'.join(tag_list))
            

if __name__ == '__main__':
    process_genia_dataset()
