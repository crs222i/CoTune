import json
import os
import subprocess


parent_env = os.environ.copy()
# 定义 10 次实验的配置（包括 data 和两组 path）
experiments = [
    {
        "data": [
            (-float('inf'), 415500, 0, 0),
            (415500, 416000, 0, 0.4),
            (416000, 420000, 0.4, 0.8),
            (420000, 440000, 0.8, 1),
            (440000, float('inf'), 1, 1)
        ],
        "path": r"..\rawData/7z/1%/results/NEW1/"  # 两个不同的 path
    },
    {
        "data": [
            (-float('inf'), 415500, 0, 0),
            (415500, 418000, 0, 0.2),
            (418000, 430000, 0.2, 0.6),
            (430000, 435000, 0.6, 1),
            (435000, float('inf'), 1, 1)
        ],
        "path": r"..\rawData/7z/1%/results/NEW2/"  # 两个不同的 path
    },
    {
        "data": [
            (-float('inf'), 415500, 0, 0),
            (415500, 417000, 0, 0.3),
            (417000, 425000, 0.3, 0.7),
            (425000, 435000, 0.7, 1),
            (435000, float('inf'), 1, 1)
        ],
        "path": r"..\rawData/7z/1%/results/NEW3/"  # 两个不同的 path
    },
    {
        "data": [
            (-float('inf'), 30000, 0, 0),
            (30000, 65000, 0, 0.3),
            (65000, 105000, 0.3, 0.8),
            (105000, 130000, 0.8, 1),
            (130000, float('inf'), 1, 1)
        ],
        "path": r"..\rawData/7z/50%/results/NEW1/"  # 两个不同的 path
    },
    {
        "data": [
            (-float('inf'), 30000, 0, 0),
            (30000, 140000, 0, 0.2),
            (140000, 400000, 0.2, 0.4),
            (400000, 430000, 0.4, 1),
            (430000, float('inf'), 1, 1)
        ],
        "path": r"..\rawData/7z/50%/results/NEW2/"  # 两个不同的 path
    },
    {
        "data": [
            (-float('inf'), 30000, 0, 0),
            (30000, 180000, 0, 0.1),
            (180000, 300000, 0.1, 0.7),
            (300000, 430000, 0.7, 1),
            (430000, float('inf'), 1, 1)
        ],
        "path": r"..\rawData/7z/50%/results/NEW3/"  # 两个不同的 path
    },
    {
        "data": [
            (-float('inf'), 280000, 0, 0),
            (280000, 350000, 0, 0.4),
            (350000, 400000, 0.4, 0.8),
            (400000, 440000, 0.8, 1),
            (440000, float('inf'), 1, 1)
        ],
        "path": r"..\rawData/7z/5%/results/NEW2/"
    },
    {
        "data": [
            (-float('inf'), 280000, 0, 0),
            (280000, 380000, 0, 0.6),
            (380000, 430000, 0.6, 0.9),
            (430000, 450000, 0.9, 1),
            (450000, float('inf'), 1, 1)
        ],
        "path": r"..\rawData/7z/5%/results/NEW3/"
    },
    {
        "data": [
            (-float('inf'), 280000, 0, 0),
            (280000, 300000, 0, 0.3),
            (300000, 420000, 0.3, 0.7),
            (420000, 430000, 0.7, 1),
            (430000, float('inf'), 1, 1)
        ],
        "path": r"..\rawData/7z/5%/results/NEW1/"
    },
    {
        "data": [
            (-float('inf'), 101600, 0, 0),
            (101600, 250000, 0, 0.3),
            (250000, 350000, 0.3, 0.7),
            (350000, 430000, 0.7, 1),
            (430000, float('inf'), 1, 1)
        ],
        "path": r"..\rawData/7z/20%/results/NEW1/"
    },
    {
        "data": [
            (-float('inf'), 101600, 0, 0),
            (101600, 150000, 0, 0.5),
            (150000, 400000, 0.5, 0.5),
            (400000, 440000, 0.5, 1),
            (440000, float('inf'), 1, 1)
        ],
        "path": r"..\rawData/7z/20%/results/NEW2/"
    },{
        "data": [
            (-float('inf'), 101600, 0, 0),
            (101600, 150000, 0, 0.6),
            (150000, 350000, 0.6, 0.8),
            (350000, 460000, 0.8, 1),
            (460000, float('inf'), 1, 1)
        ],
        "path": r"..\rawData/7z/20%/results/NEW3/"
    },
    {
        "data": [
            (-float('inf'), 421000, 0, 0),
            (421000, 421500, 0, 0.2),
            (421500, 422000, 0.2, 0.6),
            (422000, 424000, 0.6, 1),
            (424000, float('inf'), 1, 1)
        ],
        "path": r"..\rawData/7z/0.1%/results/NEW1/"
    },
    {
        "data": [
            (-float('inf'), 421000, 0, 0),
            (421000, 421700, 0, 0.1),
            (421700, 420000, 0.1, 0.5),
            (420000, 425000, 0.5, 1),
            (425000, float('inf'), 1, 1)
        ],
        "path": r"..\rawData/7z/0.1%/results/NEW2/"
    },
    {
        "data": [
            (-float('inf'), 421000, 0, 0),
            (421000, 422500, 0, 0.3),
            (422500, 423000, 0.3, 0.3),
            (423000, 423500, 0.3, 1),
            (423500, float('inf'), 1, 1)
        ],
        "path": r"..\rawData/7z/0.1%/results/NEW3/"
    },
    {
        "data": [
            (-float('inf'), 9164, 0, 0),
            (9164, 159986, 0, 0.2),
            (159986, 400000, 0.2, 0.6),
            (400000, 424000, 0.6, 1),
            (424000, float('inf'), 1, 1)
        ],
        "path": r"..\rawData/7z/90%/results/NEW1/"
    },
    {
        "data": [
            (-float('inf'), 9164, 0, 0),
            (9164, 61700, 0, 0.1),
            (61700, 300000, 0.1, 0.5),
            (300000, 425000, 0.5, 1),
            (425000, float('inf'), 1, 1)
        ],
        "path": r"..\rawData/7z/90%/results/NEW2/"
    },
    {
        "data": [
            (-float('inf'), 9164, 0, 0),
            (9164, 42500, 0, 0.3),
            (42500, 123000, 0.3, 0.3),
            (123000, 423500, 0.3, 1),
            (423500, float('inf'), 1, 1)
        ],
        "path": r"..\rawData/7z/90%/results/NEW3/"
    },
]

# 开始实验
for experiment_index, experiment in enumerate(experiments):
    # 获取 data 和 path
    data = experiment["data"]
    path = experiment["path"]
    data = json.dumps(experiment["data"])

        # 两次 30 次运行
    global_max_stagnation = 3
    parent_env.update({
        "CSV": "../csvData/7z.csv"
    })

    for run_index in range(30):
        # 更新父进程的环境变量
        parent_env.update({
            "DATA": str(data),  # 直接通过环境变量传递 data
            "RESULT_PATH": path  # 直接通过环境变量传递 path
        })

        # 执行子进程
        subprocess.run(
            [
                r"D:\论文代码\propositionSingleObj\.venv\Scripts\python.exe", "CoTune.py", str(run_index), str(global_max_stagnation)
            ],
            env=parent_env  # 使用更新后的环境变量
        )
print(f"Experiment {experiment_index + 1} completed.")