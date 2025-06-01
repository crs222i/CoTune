import copy
import json
import statistics
import sys
from typing import List
import os

from jmetal.core.problem import Problem, S
from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.core.solution import BinarySolution
from jmetal.operator import BinaryTournamentSelection, IntegerPolynomialMutation, IntegerSBXCrossover, \
    RouletteWheelSelection, BitFlipMutation, SPXCrossover
from jmetal.util.ckecking import Check
from jmetal.util.termination_criterion import StoppingByEvaluations
import pandas as pd
import numpy as np
import random
from jmetal.core.operator import Crossover, Mutation, Selection
from jmetal.util.termination_criterion import TerminationCriterion
from jmetal.util.comparator import Comparator, ObjectiveComparator
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.config import store
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.integrate import quad
from sklearn.preprocessing import MinMaxScaler
from scipy import spatial
import cProfile

global_max_stagnation = 3
path = "system"
type = "best_solution"
global_file_name = path + type
population_num = 10
history_population = set()


class Fragment:
    def __init__(self, left, right, down, up):
        self.left = left
        self.right = right
        self.down = down
        self.up = up

    def compute_objectives(self, performance) -> float:
        return self.down + (performance - self.left) / (self.right - self.left) * (self.up - self.down)


class Solution:
    def __init__(self, config_list, fitness):
        self.config_list = config_list
        self.fitness = fitness


class Proposition:
    def __init__(self, fragments: list):
        self.fragments = fragments
        self.fitness = 0

    def evaluate_proposition(self, population: List[Solution], left, right):

        # 定义待估计的点范围，从 0 到 1 的1000个点
        x_plot = np.linspace(0, 1, 200)[:, np.newaxis]

        # data为population，现在要根据不同的population计算出不同的fitness当作新的data数据
        kde_data = self.calculate_population_values(population)

        # 使用高斯核进行KDE
        kde = KernelDensity(kernel='gaussian', bandwidth=0.3)
        kde.fit(kde_data[:, np.newaxis])

        # 计算对数密度
        log_density = kde.score_samples(x_plot)
        density = np.exp(log_density)  # 转换成普通密度值

        # 找到区间 [left, right] 内的点
        mask = (x_plot[:, 0] <= right) & (x_plot[:, 0] >= left)
        x_interval = x_plot[mask]
        density_interval = density[mask]
        log_density_interval = log_density[mask]
        entropy_integral = -np.sum(density_interval * log_density_interval) * (x_interval[1] - x_interval[0])

        #print(f"数据点的熵为: {entropy_integral}")
        self.fitness = entropy_integral

    def calculate_population_values(self, population: List[Solution]):
        results = []
        for i in range(10):
            for fragment in self.fragments:
                config_value = population[i].config_list[-1]
                if fragment.left == -float('inf') and config_value <= fragment.right:
                    results.append(0)
                elif fragment.right == float('inf') and config_value >= fragment.left:
                    results.append(1)
                elif fragment.right > config_value >= fragment.left:
                    value = fragment.compute_objectives(population[i].config_list[-1])  # 假设 config[-1] 是需要的性能值
                    results.append(value)
        results.insert(0, 1)  # 在第一个位置插入0
        results.append(0)
        results.reverse()
        #print("test results:", results)
        return np.array(results)

    def ensure_continuity(self):
        """
        保证给定的Fragment列表构成连续的分段函数。
        """
        self.fragments[0].down = 0
        for i in range(1, len(self.fragments)):
            self.fragments[i].left = self.fragments[i - 1].right
            self.fragments[i].down = max(self.fragments[i - 1].up, self.fragments[i].down)
        self.fragments[-1].up = 1


# Step 1: 定义随机生成 Proposition 的方法
def generate_random_propositions(low, gap, num_proposition, num_fragments):
    propositions = []
    for i in range(num_proposition):
        left = low  # 初始 left 值为 当前种群最小值

        prev_up = 0.0  # 前一个 fragment 的 up 值，初始为 0
        fragments = [Fragment(-float('inf'), gap, 0, 0)]

        for i in range(num_fragments):
            # 生成 [8000, 19000] 范围内的 right，保证相邻 left 和 right 的连续性
            right = left + random.uniform(gap / 2, gap)  # 控制每个 fragment 的区间宽度

            # 生成 [0, 1] 范围内的 down 和 up，保证 down < up
            down = prev_up  # 当前 fragment 的 down 值等于前一个 fragment 的 up 值
            if random.random() < 0.2:
                up = down
            else:
                up = down + random.uniform(0.1, 0.25)  # 保证 up 大于 down，且随机生成

            # 保证 up 在 [0, 1] 之间
            if up > 1.0 or i == 4:
                up = 1.0
            if down > 1.0:
                down = 1.0

            # 创建 Fragment 实例并添加到 fragments 列表中
            fragment = Fragment(left, right, down, up)
            fragments.append(fragment)

            # 更新下一个 fragment 的 left 和 prev_up
            left = right
            prev_up = up

        fragments.append(Fragment(left, float('inf'), 1, 1))
        propositions.append(Proposition(fragments))
    return propositions


# Step 2: 对生成的 Proposition 进行适应度评估
def evaluate_propositions(propositions: List[Proposition], data):
    for proposition in propositions:
        proposition.evaluate_proposition(data, 0, 1)


# Step 3: 做crossover
def crossover_proposition(parents: List[Proposition]) -> List[Proposition]:
    """
    对两个Proposition进行单点交叉操作。
    """
    #print("proposition crossover")
    if len(parents) != 2:
        raise ValueError("Crossover requires exactly 2 parents.")

    parent1, parent2 = parents

    if len(parent1.fragments) < 2 or len(parent2.fragments) < 2:
        raise ValueError("Each Proposition must contain at least 2 fragments to perform crossover.")

    # 随机选择一个交叉点
    crossover_point = random.randint(1, len(parent1.fragments) - 1)
    #print("fragment len = ", len(parent1.fragments))

    matching_fragment_index = None
    for i in range(len(parent2.fragments)):
        if parent2.fragments[i].up == parent1.fragments[crossover_point].down:
            matching_fragment_index = i
            break

    # 交叉形成新的两个子代
    offspring1_fragments = parent1.fragments[:crossover_point] + parent2.fragments[matching_fragment_index:]
    offspring2_fragments = parent1.fragments[crossover_point:] + parent2.fragments[:matching_fragment_index]

    # 保证分段函数的连续性
    proposition_1 = Proposition(offspring1_fragments)
    proposition_2 = Proposition(offspring2_fragments)
    proposition_1.ensure_continuity()
    proposition_2.ensure_continuity()

    return [proposition_1, proposition_2]


# step4 mutation操作
def mutation_proposition(gap, propositions: List[Proposition]) -> List[Proposition]:
    """
    对Proposition中的一个Fragment进行变异操作。
    """
    for proposition in propositions:
        if len(proposition.fragments) < 1:
            raise ValueError("Proposition must contain at least 1 fragment to perform mutation.")

        # 随机选择一个Fragment进行变异
        if random.random() <= 0.8:
            #print("proposition mutation")
            mutate_index = random.randint(0, len(proposition.fragments) - 2)
            if mutate_index == 0:
                left_point = proposition.fragments[mutate_index].right - gap / 2
                right_point = (proposition.fragments[mutate_index + 1].right
                               + proposition.fragments[mutate_index].right) / 2
            elif mutate_index == 5:
                left_point = (proposition.fragments[mutate_index].left + proposition.fragments[mutate_index].right) / 2
                right_point = proposition.fragments[mutate_index].right + gap / 2
            else:
                left_point = proposition.fragments[mutate_index].left
                right_point = proposition.fragments[mutate_index + 1].right
            down = proposition.fragments[mutate_index].down
            up = proposition.fragments[mutate_index + 1].up
            if random.random() <= 0.1:
                tmp = down
            elif random.random() >= 0.9:
                tmp = up
            else:
                tmp = random.uniform(down, up)
            proposition.fragments[mutate_index].up = tmp
            proposition.fragments[mutate_index + 1].down = proposition.fragments[mutate_index].up

            proposition.fragments[mutate_index].right = random.uniform(left_point, right_point)
            proposition.fragments[mutate_index + 1].left = proposition.fragments[mutate_index].right
    return propositions


class PropositionSingleObjProblem(Problem):

    def __init__(self, file):
        super(PropositionSingleObjProblem, self).__init__()
        self.all_solution_set = None
        self.all_solution = None
        self.file = file
        self.original_proposition = []
        self.variable_proposition = None
        self.solution_static_list = []
        self.solution_variable_list = []
        self.parents = []
        self.independent_set = []

    # 初始化创建解决方案，随机从csv文件读取一些configurations
    def create_solution(self):
        df = pd.read_csv(self.file)
        self.all_solution = df.values.tolist()
        self.all_solution_set = set(tuple(row) for row in self.all_solution)  # 将 all_solution 转换为集合以便快速查找
        # 按列提取唯一值
        for column in df.columns[:-1]:
            self.independent_set.append(df[column].unique().tolist())
        global population_num
        # tmp_value = int(0.002 * len(self.all_solution))
        population_num = 10
        global max_budget
        max_budget = 30 * population_num
        sampled_df = df.sample(n=population_num)
        sampled_df2 = df.sample(n=population_num)
        config_list = sampled_df.values.tolist()
        config_list2 = sampled_df2.values.tolist()

        for ind, ind2 in zip(config_list, config_list2):
            # 检查ind和ind2是否在history_population集合中
            if tuple(ind) not in history_population:
                history_population.add(tuple(copy.deepcopy(ind)))  # 将深拷贝的ind作为元组存入集合
            if tuple(ind2) not in history_population:
                history_population.add(tuple(copy.deepcopy(ind2)))

        for config, config2 in zip(config_list, config_list2):
            self.solution_static_list.append(Solution(config, 0))
            self.solution_variable_list.append(Solution(config2, 0))
        return self.solution_static_list, self.solution_variable_list

    # 初始化fragment，以data的形式展现出来
    def create_fragments(self, data):
        fragment_list = []
        for left, right, down, up in data:
            current_fragment = Fragment(left, right, down, up)
            fragment_list.append(current_fragment)
        self.variable_proposition = Proposition(fragment_list)
        self.original_proposition = copy.deepcopy(self.variable_proposition)

    # 对config做适应度评估，主要参考第13个指标性能作为目标函数的参数
    def evaluate(self):
        for i in range(len(self.solution_static_list)):
            value1 = self.solution_static_list[i].config_list[-1]
            value2 = self.solution_variable_list[i].config_list[-1]
            for fragment in self.variable_proposition.fragments:
                if fragment.right >= value1 > fragment.left:
                    # population永远以original作为fitness
                    if fragment.left == -float('inf'):
                        self.solution_static_list[i].fitness = 0
                    else:
                        self.solution_static_list[i].fitness = fragment.compute_objectives(value1)
                if fragment.right >= value2 > fragment.left:
                    # population_2永远以variable作为fitness
                    if fragment.left == -float('inf'):
                        self.solution_variable_list[i].fitness = 0
                    else:
                        self.solution_variable_list[i].fitness = fragment.compute_objectives(value2)
                    break

    @property
    def name(self) -> str:
        return "Proposition Problem"

    @property
    def number_of_variables(self) -> int:
        pass

    @property
    def number_of_objectives(self) -> int:
        pass

    @property
    def number_of_constraints(self) -> int:
        return 0


max_evaluations = 100


class GeneticAlgorithm():

    def __init__(
            self,
            problem: Problem,
            population_size: int,
            offspring_population_size: int,
            mutation,
            crossover,
            selection,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
            population_generator: Generator = store.default_generator,
            population_evaluator: Evaluator = store.default_evaluator,
            solution_comparator: Comparator = ObjectiveComparator(0)
    ):
        self.population = None
        self.population_2 = None
        self.problem = problem
        self.population_size = population_size
        self.offspring_population_size = offspring_population_size
        self.mutation_operator = mutation
        self.crossover_operator = crossover
        self.solution_comparator = solution_comparator
        self.population_config = None
        self.population_proposition = None

        self.selection_operator = selection

        self.population_generator = population_generator
        self.population_evaluator = population_evaluator

        self.termination_criterion = termination_criterion
        self.data = []
        self.new_propositions = None
        self.alltime_population = []
        self.all_list = None

    #  初始化问题，得到了一个solution和fragments以及fitness
    def init_problem(self, data):
        # population是原proposition, population_2是变动proposition
        self.population, self.population_2 = self.problem.create_solution()
        self.all_list = {tuple(row[:-1]): row[-1] for row in self.problem.all_solution}
        global population_num
        self.population_size = population_num
        self.problem.create_fragments(data)
        self.problem.evaluate()
        self.iteration2()

    def evaluate_solution(self, solutions: List[Solution]) -> List[Solution]:
        for i in range(len(solutions)):
            value = solutions[i].config_list[-1]
            if value == 0:
                solutions[i].fitness = -float('inf')
            else:
                for fragment in self.problem.original_proposition.fragments:
                    if fragment.right >= value > fragment.left:
                        if fragment.left == -float('inf'):
                            solutions[i].fitness = 0
                        else:
                            solutions[i].fitness = fragment.compute_objectives(value)
                        break
        return solutions

    def evaluate_solution2(self, solutions: List[Solution]) -> List[Solution]:
        for i in range(len(solutions)):
            value = solutions[i].config_list[-1]
            if value == 0:
                solutions[i].fitness = -float('inf')
            else:
                for fragment in self.problem.variable_proposition.fragments:
                    if fragment.right >= value > fragment.left:
                        if fragment.left == -float('inf'):
                            solutions[i].fitness = 0
                        else:
                            solutions[i].fitness = fragment.compute_objectives(value)
                        break
        return solutions

    def generate_offsprings(self, population):
        global max_budget
        if isinstance(population, Solution):
            population = [population]

        success_offspring = []
        j = 0  # 当前成功生成的子代数

        while j < population_num:  # 控制生成数量和尝试上限
            #print("population_num ", population_num)
            # 随机选择父代
            parent1 = self.selection_operator.execute_tournament(population)
            remaining_population = [ind for ind in population if ind != parent1]
            parent2 = self.selection_operator.execute_tournament(remaining_population)
            parents = [parent1, parent2]

            # 交叉生成子代
            children = self.crossover_operator.execute(parents)

            # 遍历生成的子代进行变异并检查是否有效
            for child in children:
                mutated_child = self.mutation_operator.execute(child)
                result = self.all_list.get(tuple(mutated_child.config_list[:-1]))
                if result is not None:
                    # 设置子代的性能值和配置为最相似解
                    mutated_child.config_list[-1] = result
                else:
                    mutated_child.config_list[-1] = 0
                    max_budget += 1
                if tuple(mutated_child.config_list[:-1]) in history_population:
                    max_budget += 1
                else:
                    history_population.add(tuple(copy.deepcopy(mutated_child.config_list[:-1])))
                success_offspring.append(mutated_child)
                j += 1
        return success_offspring

    def handle_special_case_0(self):
        """
        Special case adjustment for fragment break‐point between
        first and second fragments, repeated until entropy decreases.
        """
        frags = self.problem.variable_proposition.fragments
        if len(frags) <= 1:
            return

        # 1) 计算原始命题的熵
        self.problem.original_proposition.evaluate_proposition(self.population_2, 0, 1)

        # 2) 反复微调，直到新熵更小或达到最大尝试次数
        max_attempts = 10
        attempts = 0
        while attempts < max_attempts:
            first_frag, second_frag = frags[0], frags[1]
            # 在 [0.5 * second.left, second.left] 范围内随机移动交点
            new_break = random.uniform(0.5 * second_frag.left, second_frag.left)
            first_frag.right = second_frag.left = new_break

            # 重新计算熵
            self.problem.variable_proposition.evaluate_proposition(self.population_2, 0, 1)
            if self.problem.variable_proposition.fitness > self.problem.original_proposition.fitness:
                print("special case 0 success")
                break
            attempts += 1

    def handle_special_case_1(self):
        """
        Special case adjustment for fragment break‐point between
        next‐to‐last and last fragments, repeated until entropy decreases.
        """
        frags = self.problem.variable_proposition.fragments
        if len(frags) <= 1:
            return

        # 1) 原始熵
        self.problem.original_proposition.evaluate_proposition(self.population_2, 0, 1)

        # 2) 重复调整
        max_attempts = 10
        attempts = 0
        while attempts < max_attempts:
            last_frag = frags[-1]
            last2_frag = frags[-2]
            # 在 [last.left, 1.5 * last.left] 范围内随机移动交点
            new_break = random.uniform(last_frag.left, 1.5 * last_frag.left)
            last2_frag.right = last_frag.left = new_break

            self.problem.variable_proposition.evaluate_proposition(self.population_2, 0, 1)
            if self.problem.variable_proposition.fitness > self.problem.original_proposition.fitness:
                print("special case 1 success")
                break
            attempts += 1

    def handle_stagnation_case(self, data):
        global global_max_data
        global global_low_data
        generation_count = 0
        max_generation = 100
        num_propositions = 10
        num_fragments = 5
        #low = self.population[int(population_num/2)].config_list[-1]
        #gap = (self.population[0].config_list[-1] - low) / 5
        low = global_low_data
        gap = (global_max_data - global_low_data) / 5
        # if self.new_propositions is None:
        #     self.new_propositions = generate_random_propositions(low, gap, num_propositions, num_fragments)
        #     evaluate_propositions(self.new_propositions, data)
        self.new_propositions = generate_random_propositions(low, gap, num_propositions, num_fragments)
        self.new_propositions.append(copy.deepcopy(self.problem.original_proposition))
        evaluate_propositions(self.new_propositions, data)
        while generation_count != max_generation:
            generation_count += 1
            selected_parents = [
                self.selection_operator.execute_rank(self.new_propositions) for ind in range(2)
            ]
            '''
            offspring_proposition = crossover_proposition(selected_parents)
            '''
            offspring_proposition = mutation_proposition(gap, selected_parents)
            evaluate_propositions(offspring_proposition, data)
            self.new_propositions.extend(offspring_proposition)
            self.new_propositions.sort(key=lambda s: s.fitness)  # 局部最优时降低选择压力，提高多样性，因此数据y轴越密集越好，那么对应fitness微分熵越低越好
            self.new_propositions = self.new_propositions[:num_propositions]
        self.problem.variable_proposition = self.new_propositions[0]
        with open(path + f"fragments_output.txt", "a") as file:
            for i, fragment in enumerate(self.problem.variable_proposition.fragments):
                file.write(f"test Fragment {i}:\n")
                file.write(f"  left: {fragment.left}\n")
                file.write(f"  right: {fragment.right}\n")
                file.write(f"  up: {fragment.up}\n")
                file.write(f"  down: {fragment.down}\n")

                # 计算两个坐标点
                left_down = (fragment.left, fragment.down)
                right_up = (fragment.right, fragment.up)
                file.write(f"  Coordinates: {left_down} to {right_up}\n")

                file.write("-" * 20 + "\n")
            file.write("\n" * 3)

    def iteration2(self):
        generation_count = 0
        stagnation_count = 0
        max_stagnation = global_max_stagnation

        w1 = 0.5
        w2 = 0.5

        pre_best_fitness1 = max(ind.fitness for ind in self.population)
        pre_best_fitness2 = max(ind.fitness for ind in self.population_2)
        current_best_fitness1 = pre_best_fitness1
        current_best_fitness2 = pre_best_fitness2
        flag = 0
        trigger = 0
        budget = len(self.population) + len(self.population_2)
        max_live = 2 * max_budget
        live = max_live

        previous_chunk = (max_budget - budget) // 10
        while not self.termination_criterion.is_met and budget < max_budget and live > 0:
            print("Generation: ", generation_count)

            # population是原proposition, population_2是变动proposition
            # 计算概率值

            if all(ind.fitness == 0 for ind in self.population):
                probability = 1.0
            # 如果 population_2 的所有 fitness 都为 1，则概率值变为 0，即只会选择原proposition生成的结果加入population
            else:
                avg_fitness1 = sum(ind.fitness for ind in self.population) / len(self.population)
                fitness_new = []
                for ind in self.population_2:
                    value = ind.config_list[-1]
                    for fragment in self.problem.original_proposition.fragments:
                        if fragment.right >= value > fragment.left:
                            if fragment.left == -float('inf'):
                                result = 0
                            else:
                                result = fragment.compute_objectives(value)
                            fitness_new.append(result)
                            break
                avg_fitness2 = sum(fitness_new) / len(fitness_new) if fitness_new else 0
                improvement1 = current_best_fitness1 - pre_best_fitness1
                improvement2 = current_best_fitness2 - pre_best_fitness2
                power1 = w1 * avg_fitness1 + w2 * improvement1
                power2 = w1 * avg_fitness2 + w2 * improvement2
                probability = power2 / (power1 + power2)

            # 记录population当前的fitness作为后续proposition变动的评估数据
            # 更新代数计数
            generation_count += 1

            current_best_fitness = self.population[0].fitness

            # 生成随机数，决定如何处理生成的后代
            if random.random() < probability or trigger != 0:
                if trigger != 0:
                    trigger -= 1
                if all(ind.fitness == 1 for ind in self.population_2):
                    self.handle_special_case_1()
                    self.population_2 = self.evaluate_solution2(self.population_2)
                success_offspring2 = self.generate_offsprings(self.population_2)
                print("success offspring", len(success_offspring2))
                offspring1 = self.evaluate_solution(success_offspring2)
                offspring2 = self.evaluate_solution2(copy.deepcopy(success_offspring2))
                live -= len(offspring1)
                budget += len(offspring1)

                self.population.extend(offspring1)
                self.population.sort(key=lambda s: s.fitness, reverse=True)  # 按适应度排序，假设越大越好
                self.population = self.population[:self.population_size]  # 取前面适应度最高的个体作为新一代种群
                self.population_2.extend(offspring2)
                self.population_2.sort(key=lambda s: s.fitness, reverse=True)  # 按适应度排序，假设越大越好
                self.population_2 = self.population_2[:self.population_size]  # 取前面适应度最高的个体作为新一代种群
            else:
                # 把变动proposition生成的后代加入population，此时认为population生成更好
                success_offspring1 = self.generate_offsprings(self.population)
                offspring1 = self.evaluate_solution(success_offspring1)
                offspring2 = self.evaluate_solution2(copy.deepcopy(success_offspring1))
                live -= len(offspring1)
                budget += len(offspring1)
                print(f"success offspring = {len(success_offspring1)}. buget = {budget}, max budget = {max_budget}")

                self.population.extend(offspring1)
                self.population.sort(key=lambda s: s.fitness, reverse=True)  # 按适应度排序，假设越大越好
                self.population = self.population[:self.population_size]  # 取前面适应度最高的个体作为新一代种群
                self.population_2.extend(offspring2)
                self.population_2.sort(key=lambda s: s.fitness, reverse=True)  # 按适应度排序，假设越大越好
                self.population_2 = self.population_2[:self.population_size]  # 取前面适应度最高的个体作为新一代种群

            pre_best_fitness1 = current_best_fitness1
            best_ind1 = max(self.population, key=lambda ind: ind.fitness)
            current_best_fitness1 = best_ind1.fitness
            current_best_config1 = best_ind1.config_list  # 你要记录的内容
            print(f"pre best:{pre_best_fitness1}, current best{current_best_fitness1}")
            pre_best_fitness2 = current_best_fitness2
            current_best_fitness2 = max(ind.fitness for ind in self.population_2)
            MAX = 0
            for ind in self.population_2:
                value = ind.config_list[-1]
                for fragment in self.problem.original_proposition.fragments:
                    if fragment.right >= value > fragment.left:
                        if fragment.left == -float('inf'):
                            result = 0
                        else:
                            result = fragment.compute_objectives(value)
                            if result > MAX:
                                MAX = result
            if current_best_fitness1 > pre_best_fitness1 or MAX > pre_best_fitness1:
                print("there is update")
                live = max_live
                flag = 1

            current_chunk = (max_budget - budget) // 10
            print(current_chunk)
            if current_chunk < previous_chunk:
                # 说明从上一个区间掉到了新的区间，如 150->140
                # 把当前 best 及其适应度写入到以 'budget{剩余预算}' 命名的文件中
                os.makedirs(path, exist_ok=True)
                with open(f"{global_file_name}_budget{(30 - previous_chunk) * 10}.txt", "a") as f:
                    f.write(str(current_best_config1) + " " + str(current_best_fitness1) + "\n")
                # 更新区间
                previous_chunk = current_chunk

            if max(ind.fitness for ind in self.population_2) == 0 and max(ind.fitness for ind in self.population) == 0 and max_stagnation != 100:
                self.handle_special_case_0()
                self.population_2 = self.evaluate_solution2(self.population_2)
            elif self.population[0].fitness == 1:
                break
            elif flag == 1:
                trigger = 0
                stagnation_count = 0  # 重置停滞计数器
                flag = 0
            else:
                stagnation_count += 1

            # 如果适应度停滞超过阈值，则执行特殊操作
            if stagnation_count > max_stagnation:
                print("there is stagnation")
                self.handle_stagnation_case(self.population_2)
                self.population_2 = self.evaluate_solution2(self.population_2)
                trigger = 2
                stagnation_count = 0  # 重置停滞计数器

            # 终止条件
            self.termination_criterion.update(self.population)
        # 返回最终的最优解, 先对2进行最终修正
        self.population_2 = self.evaluate_solution(self.population_2)
        self.population_2.sort(key=lambda s: s.fitness, reverse=True)
        for ind in self.population:
            print(f"solution{ind.config_list[-1]}, solution fitness{ind.fitness}")
        best_solution = self.population[0] if self.population[0].fitness > self.population_2[0].fitness else self.population_2[0]
        print("best solution: ", best_solution.config_list, best_solution.fitness)
        os.makedirs(path, exist_ok=True)
        max_chunk = 30
        for chunk in range((30 - previous_chunk), max_chunk + 1):  # 从下一轮开始补，例如150 -> 160 -> 170...
            os.makedirs(path, exist_ok=True)
            with open(f"{global_file_name}_budget{chunk * 10}.txt", "a") as f:
                f.write(str(best_solution.config_list) + " " + str(best_solution.fitness) + "\n")
        with open(f"{global_file_name}.txt", "a") as file:
            file.write(str(best_solution.config_list) + " " + str(best_solution.fitness) + "\n")

        return best_solution

class ConfigurationCrossover(SPXCrossover):
    def __init__(self, probability: float):
        super(SPXCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[Solution]) -> List[Solution]:
        Check.that(len(parents) == 2, "The number of parents is not two: {}".format(len(parents)))

        offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]
        rand = random.random()
        if rand <= self.probability:
            for i in range(len(parents[0].config_list) - 1):
                value_x1, value_x2 = parents[0].config_list[i], parents[1].config_list[i]

                if random.random() <= 0.5:
                    offspring[0].config_list[i] = value_x2
                    offspring[1].config_list[i] = value_x1
                else:
                    offspring[0].config_list[i] = value_x1
                    offspring[1].config_list[i] = value_x2
            offspring[0].config_list[-1] = 0
            offspring[1].config_list[-1] = 0
        return offspring


class ConfigurationMutation(BitFlipMutation):
    def __init__(self, probability: float):
        super(BitFlipMutation, self).__init__(probability=probability)

    def execute(self, solution: Solution) -> Solution:
        for i in range((len(solution.config_list) - 1)):
            rand = random.random()
            if rand <= self.probability:
                solution.config_list[i] = random.choice(algorithm.problem.independent_set[i])
        return solution


#  choose one solution everytime


class ConfigurationSelection:
    def __init__(self):
        super(RouletteWheelSelection).__init__()

    def execute(self, solutions: List[Solution]) -> Solution:
        if not solutions:
            raise Exception("The front is null")

            # 1. 过滤掉 fitness == -float('inf') 的个体
        filtered_solutions = [s for s in solutions if s.fitness != -float('inf')]
        if not filtered_solutions:
            raise Exception("过滤后没有可用的个体（所有 fitness 都是负无穷）")

        # 2. 计算调整后的适应度（示例中假设需要加上一个小扰动 epsilon）
        epsilon = 1e-6
        adjusted_fitness = [sol.fitness + epsilon * random.random() for sol in filtered_solutions]

        # 3. 计算所有适应度之和
        total_fitness = sum(adjusted_fitness)
        if total_fitness <= 0:
            raise Exception("总适应度非正，无法进行轮盘赌选择")

        # 4. 轮盘赌选择逻辑
        pick = random.uniform(0, total_fitness)
        current = 0.0
        for sol, fit in zip(filtered_solutions, adjusted_fitness):
            current += fit
            if current >= pick:
                return sol

        # 理论上不会执行到这里，但为了安全可返回最后一个个体
        return random.choice(solutions)

    def execute_tournament(self, solutions: List[Solution]) -> Solution:
        if not solutions:
            raise Exception("The front is null")

            # 1. 过滤掉 fitness == -float('inf') 的个体
        filtered_solutions = [s for s in solutions if s.fitness != -float('inf')]
        if not filtered_solutions:
            raise Exception("过滤后没有可用的个体（所有 fitness 都是负无穷）")

        # 2. 计算调整后的适应度（示例中假设需要加上一个小扰动 epsilon）
        epsilon = 1e-6
        adjusted_fitness = [sol.fitness + epsilon * random.random() for sol in filtered_solutions]

        # 3. 计算所有适应度之和
        total_fitness = sum(adjusted_fitness)
        if total_fitness <= 0:
            raise Exception("总适应度非正，无法进行轮盘赌选择")

        # 4. 轮盘赌选择逻辑
        i, j = random.sample(range(len(filtered_solutions)), 2)
        return filtered_solutions[i] if filtered_solutions[i].fitness >= filtered_solutions[j].fitness else filtered_solutions[j]

    def execute_rank(self, solutions):
        if solutions is None:
            raise Exception("The front is null")

        # 对 solutions 根据适应度从高到低进行排序
        sorted_solutions = sorted(solutions, key=lambda x: x.fitness)
        pop_size = len(sorted_solutions)

        # 计算每个个体的选择概率，线性分配权重
        selection_probs = [2 * (pop_size - rank) / (pop_size * (pop_size + 1)) for rank in range(pop_size)]

        # 使用轮盘赌法按照线性排名的概率进行选择
        selected_solution = random.choices(sorted_solutions, weights=selection_probs, k=1)[0]
        return selected_solution

    def get_name(self) -> str:
        return "Roulette wheel selection"


if __name__ == "__main__":
    problem = PropositionSingleObjProblem("Data/7z.csv")
    data = [
        [-float('inf'), 415500, 0, 0],
        [415500, 417000, 0, 0.3],
        [417000, 425000, 0.3, 0.7],
        [425000, 435000, 0.7, 1],
        [435000, float('inf'), 1, 1]
      ]

    global global_max_data  # 声明修改全局变量
    global global_low_data
    global_max_data = data[4][0]
    global_low_data = data[0][1]

    algorithm = GeneticAlgorithm(problem, population_num, population_num,
                                 mutation=ConfigurationMutation(probability=0.10),
                                 crossover=ConfigurationCrossover(probability=0.90),
                                 selection=ConfigurationSelection())
    algorithm.init_problem(data)
