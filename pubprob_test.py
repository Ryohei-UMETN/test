import numpy as np
import csv
import random
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool
import os
import datetime
import shutil

# 初期値
num_agent = 400  # エージェント数
num_periods = int(1000)  # ピリオド数
cost_of_cooperation = 1  # 協力するコスト
num_generations = 1000  # ジェネレーション数
mutation_rate = 0.01  # 突然変異確率
simulation = 50  # シミュレーション回数

# エージェントクラスの定義
class Agent:
    def __init__(self, id):
        self.id = id  # エージェントID
        self.payoff = 0  # 利得
        self.norm = np.random.choice(['G', 'B'], size=4)  # 規範
        self.reputation = ['G'] * num_agent  # 評価配列（初期値は全てGood）
        self.reputation[self.id] = 'G'  # 自分自身をGoodと評価する

    def decide_action(self, recipient_reputation):
        if ''.join(self.norm) == 'GGGG':
            return 'C'
        elif ''.join(self.norm) == 'BBBB':
            return 'D'
        return 'C' if recipient_reputation == 'G' else 'D'  # 通常の行動

    def update_reputation(self, donor_action, recipient_reputation):
        if recipient_reputation == 'G' and donor_action == 'C':
            return self.norm[0]
        elif recipient_reputation == 'G' and donor_action == 'D':
            return self.norm[1]
        elif recipient_reputation == 'B' and donor_action == 'C':
            return self.norm[2]
        elif recipient_reputation == 'B' and donor_action == 'D':
            return self.norm[3]

# 公的機関クラスの定義
class PublicInstitution:
    def __init__(self, norm):
        self.norm = norm  # 公的規範
        self.reputation = ['G'] * num_agent  # 公的評価配列（初期値は全てGood）

    def update_reputation(self, donor_action, recipient_reputation):
        if recipient_reputation == 'G' and donor_action == 'C':
            return self.norm[0]
        elif recipient_reputation == 'G' and donor_action == 'D':
            return self.norm[1]
        elif recipient_reputation == 'B' and donor_action == 'C':
            return self.norm[2]
        elif recipient_reputation == 'B' and donor_action == 'D':
            return self.norm[3]

# 役割の割り当て
def assign_roles(num_agent):
    agents = list(range(num_agent))
    random.shuffle(agents)
    return [(agents[i], agents[(i + 1) % num_agent]) for i in range(num_agent)]


def save_simulation_log(script_path="your_script_name.py", log_dir="logs"):
    # 現在時刻を取得
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    # ログディレクトリ作成（存在しなければ）
    os.makedirs(log_dir, exist_ok=True)

    # ログファイル名
    log_filename = f"simulation_log_{timestamp}.txt"
    log_path = os.path.join(log_dir, log_filename)

    # スクリプト内容をコピーして保存
    with open(script_path, "r") as src, open(log_path, "w") as dst:
        dst.write(f"# Simulation started at: {now.isoformat()}\n\n")
        dst.write(src.read())

    print(f"ログファイル生成: {log_path}")


# シミュレーション実行関数
def run_simulation(params):
    error_rate_action, error_rate_evaluation, error_rate_public_evaluation, benefit_of_cooperation, probability, public_norm, sim = params
    agents = [Agent(i) for i in range(num_agent)]
    public_institution = PublicInstitution(public_norm)

    cooperation_rates = []  # 協力率を記録するリスト
    norm_distribution = []  # 規範の分布を記録するリスト

    for generation in range(num_generations):
        cooperation_count = 0
        interaction_count = 0

        for period in range(num_periods):
            pairs = assign_roles(num_agent)

            for donor_id, recipient_id in pairs:
                donor = agents[donor_id]
                recipient = agents[recipient_id]

                action = donor.decide_action(donor.reputation[recipient_id])
                if np.random.rand() < error_rate_action:
                    action = 'D' if action == 'C' else 'C'

                if period >= 0.8 * num_periods:
                    interaction_count += 1
                if action == 'C':
                    donor.payoff -= cost_of_cooperation
                    recipient.payoff += benefit_of_cooperation
                    if period >= 0.8 * num_periods and action == 'C':
                        cooperation_count += 1

                # 評判の更新（このアクションごとに即時に更新する）
                # 公的機関の評価
                public_institution.reputation[donor_id] = public_institution.update_reputation(
                    action, public_institution.reputation[recipient_id])
                if np.random.rand() < error_rate_public_evaluation:
                    public_institution.reputation[donor_id] = (
                        'B' if public_institution.reputation[donor_id] == 'G' else 'G'
                    )

                # 各エージェントによる評価
                for evaluator_id in range(num_agent):
                    if evaluator_id == donor_id:
                        continue
                    if np.random.rand() < probability:
                        evaluator = agents[evaluator_id]
                        evaluator.reputation[donor_id] = evaluator.update_reputation(
                            action, evaluator.reputation[recipient_id])
                        if np.random.rand() < error_rate_evaluation:
                            evaluator.reputation[donor_id] = (
                                'B' if evaluator.reputation[donor_id] == 'G' else 'G'
                            )
                    else:
                        agents[evaluator_id].reputation[donor_id] = public_institution.reputation[donor_id]

        cooperation_rate = cooperation_count / interaction_count
        cooperation_rates.append(cooperation_rate)
        norm_distribution.append([agent.norm for agent in agents])

        payoffs = np.array([agent.payoff for agent in agents])
        if np.all(payoffs == payoffs[0]):
            weights = np.ones_like(payoffs) / len(payoffs)
        else:
            payoffs -= payoffs.min()
            payoffs = np.square(payoffs)
            weights = payoffs / payoffs.sum()

        new_agents = []
        for _ in range(num_agent):
            parents = np.random.choice(agents, size=2, p=weights)
            new_norm = []
            for i in range(4):
                inherited = random.choice([parents[0].norm[i], parents[1].norm[i]])
                if np.random.rand() < mutation_rate:
                    mutated = 'B' if inherited == 'G' else 'G'
                    new_norm.append(mutated)
                else:
                    new_norm.append(inherited)
            new_agent = Agent(len(new_agents))
            new_agent.norm = new_norm
            new_agents.append(new_agent)
        agents = new_agents

        for agent in agents:
            agent.payoff = 0
            agent.reputation = ['G'] * num_agent
            agent.reputation[agent.id] = 'G'
        public_institution.reputation = ['G'] * num_agent

        file_prefix = f"{num_agent}_{public_norm}_probability{probability}_action_error{error_rate_action}_evaluate_error{error_rate_evaluation}_public_error{error_rate_public_evaluation}_benefit{benefit_of_cooperation}_{sim+1}"
        file_prefix_without_sim = f"{num_agent}_{public_norm}_probability{probability}_action_error{error_rate_action}_evaluate_error{error_rate_evaluation}_public_error{error_rate_public_evaluation}_benefit{benefit_of_cooperation}"
      
        
        norm_file = f"norm_distribution{file_prefix}.csv"
        with open(norm_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Generation"] + [f"Agent_{i}" for i in range(num_agent)])
            for gen_idx, norms in enumerate(norm_distribution):
                writer.writerow([gen_idx] + [''.join(norm) for norm in norms])

        # 新ファイル名（simを含めない）
        coop_file = f"cooperation_rates{file_prefix_without_sim}.csv"
        # ファイルが既に存在するかを確認して、追記 or 新規作成
        write_header = not os.path.exists(coop_file)
        with open(coop_file, "a", newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Simulation", "Generation", "CooperationRate"])
            for gen_idx, coop_rate in enumerate(cooperation_rates):
                writer.writerow([sim + 1, gen_idx + 1, coop_rate])


# メイン関数
def main():
    save_simulation_log(script_path=__file__) 
    error_rates_action = [0, 0.001]  # アクションエラー率の設定
    error_rates_evaluation = [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01]
    benefit_values = [1, 2, 3, 4, 5]  # 協力の利益
    probability_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # ネットワークの次数
    public_norms = ['GBBG', 'GBGB', 'GBGG', 'GBBB']  # 公的規範
    params = []

    for error_rate_action in error_rates_action:
        for error_rate_evaluation in error_rates_evaluation:
            for error_rate_public_evaluation in error_rates_evaluation:
                for benefit_of_cooperation in benefit_values:
                    for probability in probability_values:
                        for public_norm in public_norms:
                            for sim in range(simulation):
                                params.append((error_rate_action, error_rate_evaluation, error_rate_public_evaluation, benefit_of_cooperation, probability, public_norm, sim))

    with Pool() as pool:
        list(tqdm(pool.imap_unordered(run_simulation, params), total=len(params)))

if __name__ == "__main__":
    main()
