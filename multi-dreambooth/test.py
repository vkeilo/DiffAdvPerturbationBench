concepts_list = [
    {
        "instance_prompt":      "photo of zwx dog",
        "class_prompt":         "photo of a dog",
        "instance_data_dir":    "/content/data/zwx",
        "class_data_dir":       "/content/data/dog"
    },
    {
        "instance_prompt":      "photo of ukj person",
        "class_prompt":         "photo of a person",
        "instance_data_dir":    "/content/data/ukj",
        "class_data_dir":       "/content/data/person"
    }
]

# `class_data_dir` contains regularization images
import json
import os
# for c in concepts_list:
#     os.makedirs(c["instance_data_dir"], exist_ok=True)

with open("concepts_list.json", "w") as f:
    json.dump(concepts_list, f, indent=4)