import numpy as np
import json

from utils.visualizer import Trajectory, Trajectories


class Metric:
    def __init__(self) -> None:
        self.total_reward = 0
        self.ep_length = 0
        self.basic_reward = 0
        self.addon_reward = 0
        self.addon_metric = []

    def reset(self) -> None:
        self.total_reward = 0
        self.ep_length = 0
        self.basic_reward = 0
        self.addon_reward = 0
        self.addon_metric = []

    def update(self, info) -> None:
        self.total_reward += info[0]["total_reward"]
        self.ep_length += 1
        self.basic_reward += info[0]["basic_reward"]
        self.addon_reward += info[0]["addon_reward"]
        self.addon_metric.append(info[0]["addon_metric"])

class Metrics:
    def __init__(self) -> None:
        self.total_rewards = []
        self.ep_lengths = []
        self.basic_rewards = []
        self.addon_rewards = []
        self.addon_metrics = []

    def reset(self) -> None:
        self.total_rewards = []
        self.ep_lengths = []
        self.basic_rewards = []
        self.addon_rewards = []
        self.addon_metrics = []

    def update(self, metric: Metric, env_id=None) -> None:
        self.total_rewards.append(metric.total_reward)
        self.ep_lengths.append(metric.ep_length)
        self.basic_rewards.append(metric.basic_reward)
        self.addon_rewards.append(metric.addon_reward)
        self.addon_metrics.append(np.mean(metric.addon_metric).item())  
        # self.addon_metrics.append(np.mean(metric.addon_metric).item())

    def extract(self, extra_data=None) -> None:
        data = {}
        if extra_data is not None:
            for key, value in extra_data.items():
                data[key] = value
        data["ep_lengths_mean"] = np.mean(self.ep_lengths)
        data["ep_lengths_std"] = np.std(self.ep_lengths)
        data["total_rewards_mean"] = np.mean(self.total_rewards)
        data["total_rewards_std"] = np.std(self.total_rewards)
        data["basic_rewards_mean"] = np.mean(self.basic_rewards)
        data["basic_rewards_std"] = np.std(self.basic_rewards)
        data["addon_rewards_mean"] = np.mean(self.addon_rewards)
        data["addon_rewards_std"] = np.std(self.addon_rewards)
        data["addon_metrics_mean"] = np.mean(self.addon_metrics)
        data["addon_metrics_std"] = np.std(self.addon_metrics)

        return data

    def summary(self) -> None:
        print(f"Total Reward: {np.mean(self.total_rewards):.2f} +/- {np.std(self.total_rewards):.2f}")
        print(f"Episode Length: {np.mean(self.ep_lengths):.2f} +/- {np.std(self.ep_lengths):.2f}")
        print(f"Basic Reward: {np.mean(self.basic_rewards):.2f} +/- {np.std(self.basic_rewards):.2f}")
        print(f"Add on Reward: {np.mean(self.addon_rewards):.2f} +/- {np.std(self.addon_rewards):.2f}")
        print(f"Add on Matrix: {np.mean(self.addon_metrics):.2f} +/- {np.std(self.addon_metrics):.2f}")
        print("")

class EvalLog:
    def __init__(self) -> None:
        self.total_metrics = Metrics()
        self.success_metrics = Metrics()
        self.fail_metrics = Metrics()

        self.success_rate = 0

    def reset(self) -> None:
        self.total_metrics.reset()
        self.success_metrics.reset()
        self.fail_metrics.reset()

    def save(self, data: dict, log_path: str) -> None:
        with open(log_path,"w", encoding='utf-8') as file:
            data["total"] = self.total_metrics.extract()
            data["success"] = self.success_metrics.extract()
            data["fail"] = self.fail_metrics.extract()
            file.write(json.dumps(data, ensure_ascii=False, indent=4))

    def summary(self) -> None:
        print("############################################")
        self.success_rate = len(self.success_metrics.total_rewards) / len(self.total_metrics.total_rewards)
        print(f"Success rate: {(100 * self.success_rate):.2f}%")
        # print(f"Planning Freq.: {np.mean(planning_frequency):.2f} +/- {np.std(planning_frequency):.2f}")
        print("-------------------TOTAL-------------------")
        self.total_metrics.summary()

        print("------------------SUCCESS------------------")
        self.success_metrics.summary()

        print("--------------------FAIL-------------------")
        self.fail_metrics.summary()
