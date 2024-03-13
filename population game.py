import numpy as np
import random
from matplotlib import pyplot as plt
from typing import List
from matplotlib.animation import FuncAnimation



class StateAgent():
    def __init__(self):
        self.resource :float = 1
        self.death: bool = False
        self.replication: bool = False
    
    def feed(self, resource: float) -> tuple:
        self.death = False
        self.replication = False
        self.resource += resource
       
        if self.resource < 1:
            resource_consumed = self.resource        
        elif self.resource >= 1:
            resource_consumed = random.uniform(1, self.resource)
        self.resource -= resource_consumed
        
        if resource_consumed < 1:
            death_prob = 1-resource_consumed
            if random.random() < death_prob:
                self.death = True
        elif resource_consumed > 1:
            replication_prob = resource_consumed - 1 #certain replication if resource consumed bigger or equal to 2 unit
            death_prob = (1/9)*(resource_consumed-1)   #certain death if resource consumed bigger or equal to 10 unit, aka 10* the gauranteed survival amount
            if random.random() < replication_prob:
                self.replication = True
            if random.random() < death_prob:     #no replication if death occured by overconsumption
                self.death = True
                self.replication = False
        if resource_consumed == 1:
            self.death = False
            self.replication = False

        

        return self.death, self.replication
            


class Environment():
    def __init__(self, n_agents: int, q_resources: float, equality: bool = False):

        self.agents = [StateAgent() for _ in range(n_agents)]
        self.resource_available = q_resources
        self.n_alive = n_agents
        self.equality = equality    #if True, resources are distributed equally, else, resources are distributed randomly but fairly

    def distribute_resource(self):
        points = [random.uniform(0, self.resource_available) for _ in range(self.n_alive-1)]
        points.extend([0, self.resource_available])    
        points.sort()
        segments = [points[i+1] - points[i] for i in range(len(points)-1)]
        return segments
    
    def fairly_distribute_resource(self):
        base_share = self.resource_available / self.n_alive  # Equal share for each agent
        deviation_range = base_share * 0.1
        segments = [base_share + random.uniform(-deviation_range, deviation_range) for _ in range(self.n_alive)]
        total_distributed = sum(segments)
        adjustment_factor = self.resource_available / total_distributed
        segments = [segment * adjustment_factor for segment in segments]
        return segments
    
    def next_iteration(self):
        random.shuffle(self.agents)
        if self.equality == True:
            resource_distributed = self.fairly_distribute_resource()
        else:
            resource_distributed = self.distribute_resource()
        for resource, agent in zip(resource_distributed, self.agents):
            death, replication = agent.feed(resource)
        # with ProcessPoolExecutor(max_workers=len(self.agents)) as executor:
        #     results = executor.map(lambda agent, resource: agent.feed(resource), self.agents, resource_distributed)
        #     for agent, (death, replication) in zip(self.agents, results):
            if death:
                self.n_alive -= 1
                self.agents.remove(agent)
            if replication:
                self.n_alive += 1
                self.agents.append(StateAgent())
        self.n_alive = len(self.agents)
        resource_held_sorted_ascending = sorted((agent.resource for agent in self.agents), reverse=False)
        cumulative_resource_held: list[float] = np.cumsum([resource for resource in resource_held_sorted_ascending]).tolist()
        cumulative_resource_held_percentage = [100*cumulative_resource_held[i]/max(cumulative_resource_held) for i in range(len(cumulative_resource_held))]
        print(self.n_alive)
        return cumulative_resource_held_percentage
    
    def run(self, n_iterations: int) -> List[List[float]]:
        simulation_data: list[list] = []
        for _ in range(n_iterations):
            n = self.next_iteration()
            simulation_data.append(n)
        return simulation_data


def calculate_gini(percentages):
    # Ensure the list is sorted
    percentages = sorted(percentages)
    # Calculate cumulative sums
    cum_pop = [i / len(percentages) for i in range(1, len(percentages) + 1)]
    cum_res = [sum(percentages[:i]) / sum(percentages) for i in range(1, len(percentages) + 1)]
    # Calculate the area under the Lorenz curve using the trapezoidal rule
    A = 0
    for i in range(1, len(cum_pop)):
        A += 0.5 * (cum_pop[i] - cum_pop[i-1]) * (cum_res[i] + cum_res[i-1])
    # Calculate and return the Gini coefficient
    G = 1 - 2 * A
    return G

def simulate(n_agents: int, q_resources: list[float], n_iterations: int, equality_mode: bool = False) -> list:
    results = []
    for q in q_resources:
        simulation = Environment(n_agents, q, equality = equality_mode)
        result =  simulation.run(n_iterations)
        results.append(result)
    for result in results:
        gini_coefficients = [calculate_gini(percentages) for percentages in result]
        fig, ax = plt.subplots()
        line, = ax.plot([], [], lw=2)
        # Setting the limits of x and y axes
        ax.set_xlim(0, max(len(r) for r in result))  # Assuming a maximum of 100 agents for visualization clarity
        ax.set_ylim(0, max(max(result), default=0))  # Adjust based on your maximum cumulative resource, adding some padding
        ax.set_xlabel('Number of Agents Alive')  # X-axis title
        ax.set_ylabel('% Cumulative Resource Held')  # Y-axis title
        ax.set_title('Simulation of Resource Distribution Over Time')
        def init():
            line.set_data([], [])
            return line,
        def update(frame):
            y = result[frame]
            x = range(len(y))
            line.set_data(x, y)
            return line,
        ani = FuncAnimation(fig, update, frames=range(len(result)),init_func=init, blit=True, repeat=False, interval=300)
        plt.show()
        plt.plot(gini_coefficients)
        plt.show()





simulate(100, [10, 30, 50, 70, 90], 500, equality_mode = False)
simulate(100, [100, 200, 300, 400, 500], 500, equality_mode = False)
simulate(100, [600,700,800,900,1000], 500, equality_mode = False)
simulate(100, [1100, 1200, 1300, 1400, 1500], 500, equality_mode = False)

#critical points
simulate(100, [900, 920,940,960,980,1000], 500, equality_mode = False)

        



simulate(100, [10, 30, 50, 70, 90], 500, equality_mode = True)
simulate(100, [100, 200, 300, 400, 500], 500, equality_mode = True)
simulate(100, [600,700,800,900,1000], 500, equality_mode = True)
simulate(100, [1100, 1200, 1300, 1400, 1500], 500, equality_mode = True)

