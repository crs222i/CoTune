# CoTune: Co-evolutionary Configuration Tuning
This repository contains the data and code for the following paper:
**Co-evolutionary Configuration Tuning**
## Introduction
To automatically tune configurations for the best possible system performance (e.g., runtime or throughput), much work has focused on designing intelligent heuristics in a tuner. However, existing tuner designs have mostly ignored the presence of complex performance requirements (e.g., “the latency shall ideally be 2 seconds”), simply assuming that better performance is always preferable. This oversight not only wastes valuable requirement information but may also expend extensive resources to achieve minimal gains. Moreover, prior studies have shown that directly incorporating a strict requirement as an optimization objective can harm convergence or cause premature convergence due to widely varying satisfaction levels.

In this paper, we propose CoTune, a tool that integrates a target performance requirement via co-evolution. CoTune’s core idea is to co-evolve an auxiliary requirement alongside the configurations. When the main requirement becomes ineffective or misleading, the auxiliary requirement intervenes to guide the search, ensuring that tuning remains both requirement-driven and robust against harmful effects.
## Systems 

| System      | Benchmark             | Domain            | Lang.   | Perf. Metric       | `|B|/|N|` | S_space        |
|-------------|-----------------------|-------------------|---------|--------------------|-----------|----------------|
| 7z          | A 3 GB directory      | File Compressor   | C++     | Runtime (ms)       | 11/3      | 4.39×10⁵       |
| KANZI       | Default benchmark     | File Compressor   | Java    | Runtime (ms)       | 31/0      | 5.36×10⁸       |
| ExaStencils | Default benchmark     | Code Generator    | Scala   | Runtime (ms)       | 7/5       | 6.55×10⁸       |
| Apache      | Web server benchmark  | Web Server        | C       | Throughput (req/s) | 14/2      | 3.28×10⁴       |
| SQLite      | Default benchmark     | Database Engine   | Various | Latency (ms)       | 39/0      | 5.50×10¹¹      |
| DConvert    | Default images        | Image Scaling     | Java    | Runtime (s)        | 17/1      | 2.62×10⁵       |
| DeepArch    | UCR Archive dataset   | AI Tool           | Python  | Runtime (min)      | 12/0      | 4.10×10³       |
| Jump3r      | Jump3r codebase       | Static Analysis   | Java    | Runtime (ms)       | 37/0      | 6.87×10¹⁰      |
| HSMGP       | V-cycle solver bench  | Multigrid Solver  | C++     | Runtime (ms)       | 11/3      | 1.00×10⁵       |

## Structure

- Code\CoTune.py =>  The reproduction code of CoTune
- Code\proposition.JSON => Different optional initial propositions for different systems at different percentages. The percentage refers to the proportion of non-zero numbers in this csvData set
- Code\restart_sys => In order to test the data structure of different systems in different situations on a large scale
- rawData => Experimental results
- csvData => Datasets
- requirements.txt => Essential requirments need to be installed

## Quick Start

- Python 3.8+

To run the code, and install the essential requirements:

```
pip install -r requirements.txt
```

`Set line 213 r"yourProject\.venv\Scripts\python.exe" with your local python installation address.`

And you can run the below code to have a quick start:

```
cd ./Code
python3 restart_sys.py
```

Note: If you have your own dataset, put the CSV file in "csvData", and set your own propositions in restart_sys.py and based on your CSV data. Finally, add it to the datasets list and run CoTune.

## Compared with GA

- [GA](https://github.com/jMetal/jMetalPy): a genetic algorithm for optimal configurations using natural selection and cross-variance heuristics.

Note: We used two objective values for guidance. One is satisfaction, the other is original performance value.

## Compared with State-of-the-Art Tuners

- [SMAC](https://github.com/automl/SMAC3): a sequential model-based optimizer that handles categorical parameters by building a random forest to identify promising configurations in the algorithm configuration space.
- [FLASH](https://github.com/FlashRepo/Flash-SingleConfig): a sequential model-based method for single-objective configuration optimization that leverages prior knowledge of the configuration space to choose the next configuration, reducing the number of measurements needed.
- [BOCA](https://github.com/BOCA313/BOCA): an automatic compiler tuning method based on bayesian optimization, which designs novel search strategies by approximating the objective function using a tree-based model.

Note: Like GA, the SOTA methods can also use two different bootstrapping methods. One is satisfaction, the other is original performance value.

## RQ Reproduction

- **RQ1 Co-evolution necessity**: To measure the necessity of co-evolution in our method, you can directly run [Quick Start](#quick-start). The other GA methods being compared are described in [Compared with GA](#compared-with-ga).

- **RQ2 State-of-the-art tuners**: Compared with the state of the art algorithm. The other SOTA methods being compared are described in [Compared with State-of-the-Art Tuners](#compated-with-state-of-the-art-tuners).

- **RQ3 Ablation**:  Compare the differences when the key component is uesd and others are not used:

  1. Component 0:  `Comment out line 607, 687-689 in Code\CoTune.py.`
  2. Component 1: `Comment out line 673, 687-689 in Code\CoTune.py.`
  3. Component 2: `Comment out line 607, 673 in Code\CoTune.py.`

  The above three methods test the individual functions of component 0, 1, and 2 respectively.

- **RQ4 Sensitivity**: Analyze the sensitivity of the key parameter k, and set it to 3, 5, 7 or 10:

​	`Set line 196 in Code\restart_sys.py with 'global_max_stagnation = 5' .`

## RQ Supplementary

RQ_supplementary contains the specific supplementary files for RQ3.
