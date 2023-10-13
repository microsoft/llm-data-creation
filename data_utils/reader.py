import os

from data_utils.utils import Split, read_jsonl


class Reader:
    def __init__(
        self,
        data_dir: str,
        data_name: str,
        mode: Split,
        setting: str,
        start: int = 0,
        end: int = 10,
    ):
        self.data_dir = data_dir
        self.data_name = data_name
        self.instances = []
        self.setting = setting

        if mode == Split.VAL:
            self.instances = read_jsonl(os.path.join(self.data_dir, self.data_name) + "/dev.jsonl")
        elif mode == Split.TEST:
            self.instances = read_jsonl(os.path.join(self.data_dir, self.data_name) + "/test.jsonl")
        else:
            if setting == "naive":
                self.instances = read_jsonl(
                    os.path.join(self.data_dir, self.data_name) + "/train.jsonl"
                )
            else:
                self.instances = read_jsonl(
                    os.path.join(self.data_dir, self.data_name) + f"/train_{setting}.jsonl"
                )

            train_size = len(self.instances)

            start_point = train_size * start / 10
            end_point = train_size * end / 10

            self.instances = self.instances[int(start_point) : int(end_point)]

    def load(self):
        final_instances = []

        for x in self.instances:
            final_instances.append(x)

        return final_instances
