import json
from campus_demand import evaluate

if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
