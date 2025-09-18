import json
import os
from copy import deepcopy
from optimizer_module import BlockSchedulerOptimizer, MockRLAgent, TrainAction

# ------------------------
# 1. Simple Simulator Update
# ------------------------
def update_simulator_state(current_state: dict, optimized_schedule: dict) -> dict:
    """
    Apply optimizer decisions to update the simulator state.
    """
    new_state = deepcopy(current_state)
    
    # Update blocks and train positions
    for action in optimized_schedule['train_actions']:
        train_id = action['train_id']
        train_info = new_state['trains'][train_id]
        
        if action['action_type'] == 'move_to_block':
            old_block = train_info.get('current_block')
            new_block = action['target_block']
            
            # Free old block
            if old_block:
                new_state['blocks'][old_block]['is_free'] = True
                new_state['blocks'][old_block]['current_train'] = None
            
            # Occupy new block
            train_info['current_block'] = new_block
            train_info['next_block'] = get_next_block(train_info)
            new_state['blocks'][new_block]['is_free'] = False
            new_state['blocks'][new_block]['current_train'] = train_id
            
            # Update train state
            train_info['state'] = 'moving'
        elif action['action_type'] == 'hold':
            train_info['state'] = 'waiting'
    
    # Advance simulation time
    new_state['time'] += 30  # assume 30 sec per step
    
    return new_state

def get_next_block(train_info: dict) -> str:
    """Get the next block in the train's route"""
    route = train_info.get('route', [])
    index = train_info.get('route_index', 0)
    if index + 1 < len(route):
        train_info['route_index'] = index + 1
        return route[index + 1]
    return None

# ------------------------
# 2. Print Simulation State
# ------------------------
def print_simulation_state(state: dict):
    """
    Print table of train positions, state, and block occupancy.
    """
    print(f"\nTime: {state['time']} seconds")
    print("Block Occupancy:")
    print("Block | Train")
    for block_id, block_info in state['blocks'].items():
        train = block_info['current_train'] or "-"
        print(f"{block_id:5} | {train}")
    
    print("\nTrain States:")
    print("Train | State   | Current Block | Next Block | Delay")
    for train_id, train_info in state['trains'].items():
        print(f"{train_id:5} | {train_info['state']:7} | {train_info.get('current_block','-'):13} | "
              f"{train_info.get('next_block','-'):10} | {train_info.get('current_delay',0):5}")
    
    # Calculate congestion level
    total_blocks = len(state['blocks'])
    occupied_blocks = sum(1 for b in state['blocks'].values() if not b['is_free'])
    congestion = occupied_blocks / total_blocks
    print(f"\nCongestion Level: {congestion:.2f}")

# ------------------------
# 3. Load Initial Simulator State
# ------------------------
# Get the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct full path to the JSON file
simulator_file = os.path.join(script_dir,  "simulator", "simulator_state.json")
with open("simulator_state.json") as f:
    current_state = json.load(f)

# ------------------------
# 4. Initialize RL Agent and Optimizer
# ------------------------
rl_agent = MockRLAgent()
optimizer = BlockSchedulerOptimizer(rl_agent=rl_agent)

# ------------------------
# 5. Run Multi-Step Simulation
# ------------------------
num_steps = 5  # number of simulation steps
for step in range(num_steps):
    print(f"\n=== Simulation Step {step + 1} ===")
    
    # Run optimizer
    optimized_schedule = optimizer.optimize(current_state)
    
    # Print optimizer decision
    print("\nOptimizer Actions:")
    for act in optimized_schedule['train_actions']:
        print(f"Train {act['train_id']} -> {act['action_type']} -> {act.get('target_block', '-')}")
    
    # Update simulator state
    current_state = update_simulator_state(current_state, optimized_schedule)
    
    # Print updated simulation state
    print_simulation_state(current_state)
