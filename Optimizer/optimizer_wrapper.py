import json
import os
from optimizer_module import BlockSchedulerOptimizer, MockRLAgent  # import your classes

def load_simulator_state(filename: str) -> dict:
    """Load simulator state from JSON file located in script folder"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)
    with open(filepath, 'r') as f:
        state = json.load(f)
    return state


def save_optimizer_output(output: dict, filename: str):
    """Save optimizer decision log to JSON file"""
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

def main(simulator_file: str, output_file: str):
    # 1. Load current simulator state
    current_state = load_simulator_state(simulator_file)

    # 2. Initialize RL agent (replace MockRLAgent with your trained agent if available)
    rl_agent = MockRLAgent()

    # 3. Initialize optimizer
    optimizer = BlockSchedulerOptimizer(rl_agent=rl_agent)

    # 4. Run optimization
    optimized_schedule = optimizer.optimize(current_state)

    # 5. Save JSON output
    save_optimizer_output(optimized_schedule, output_file)

    print(f"Optimization complete. Output saved to {output_file}")
    print(json.dumps(optimized_schedule, indent=2))


# Get the folder of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(script_dir, "simulator_state.json")

with open(filename, 'r') as f:
    current_state = json.load(f)

if __name__ == "__main__":
    # Example usage
    simulator_input_file = "simulator_state.json"  # simulator will produce this
    optimizer_output_file = "optimizer_schedule.json"

    main(simulator_input_file, optimizer_output_file)
