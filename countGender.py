import json

datasets = ["TinyStress"]#, "StressTest", "StressPresso", "Emphassess"]

for dataset in datasets:
    total = 0
    female = 0
    gender = ""
    with open(f"/datas/store162/annhung/Qwen3-SLU/data-json/{dataset}/train.jsonl", 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        gender = result.get("gender","")
        if(gender == "female"):
            total += 1
            female += 1
        elif(gender == "male"):
            total += 1
        # print(f"result: {result}")
        # print(isinstance(result, dict))
    print(f"{dataset}:{female}/{total} = ({female/total})")