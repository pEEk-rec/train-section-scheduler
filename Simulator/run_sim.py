"""# run_sim.py
import json
import argparse
from simulator import Simulator  # assumes simulator.py in same folder

def serialize_sim(sim):
    events = []
    for e in sim.logs:
        events.append({
            "time": e.time,
            "event_type": e.event_type.value,
            "train_id": e.train_id,
            "section_id": e.section_id,
            "details": e.details
        })
    trains = {}
    for tid, tr in sim.trains.items():
        trains[tid] = {
            "state": tr.state,
            "route_index": tr.route_index,
            "timeline": [
                {"section_id": ts.section_id, "entry_time": ts.entry_time, "exit_time": ts.exit_time}
                for ts in tr.timeline
            ]
        }
    return {"events": events, "trains": trains, "metrics": sim.compute_metrics()}

def run(scenario_path, out_path, until_time=None, decision_interval=None):
    with open(scenario_path) as f:
        scenario = json.load(f)
    sim = Simulator(decision_interval=decision_interval or scenario.get("global", {}).get("decision_interval_s", 5.0))
    sim.load_scenario(scenario)
    sim.run(until_time=until_time)
    out = serialize_sim(sim)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved", out_path)
    print("Metrics:", out["metrics"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="scenarios/sample.json")
    parser.add_argument("--out", default="logs/run1.json")
    parser.add_argument("--until", type=float, default=300.0)
    parser.add_argument("--decision_interval", type=float, default=None)
    args = parser.parse_args()
    run(args.scenario, args.out, until_time=args.until, decision_interval=args.decision_interval)
*/"""
import json
from simulator import Simulator

def run_scenario():
    # Hardcoded file path
    input_file = "scenarios/scenario2.json"
    output_file = "logs/run1.json"

    # Load external scenario JSON
    with open(input_file, "r") as f:
        scenario = json.load(f)

    # Create and run simulation
    sim = Simulator(decision_interval=5.0)
    sim.load_scenario(scenario)
    sim.run(until_time=300.0)

    # Save logs
    sim.save_logs(output_file)

    # Print metrics
    metrics = sim.compute_metrics()
    print("\nSimulation Results:")
    print(f"Completed trains: {metrics['completed_trains']}/{metrics['total_trains']}")
    print(f"Completion rate: {metrics['completion_rate']:.2%}")
    print(f"Average delay: {metrics['average_delay']:.2f} seconds")
    print(f"Total simulation time: {metrics['total_simulation_time']:.2f} seconds")

    # Print last 10 events
    print("\nEvent Log (last 10 events):")
    for log in sim.logs[-10:]:
        print(f"{log.time:.2f}s: {log.event_type.value} - Train: {log.train_id}, Section: {log.section_id}")

# Run directly
if __name__ == "__main__":
    run_scenario()
