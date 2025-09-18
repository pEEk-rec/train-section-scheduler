import heapq
import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from abc import ABC, abstractmethod

class EventType(Enum):
    TRAIN_ARRIVAL_AT_SECTION = "train_arrival_at_section"
    TRAIN_LEAVES_SECTION = "train_leaves_section"
    TRAIN_ARRIVAL_AT_PLATFORM = "train_arrival_at_platform"
    DISPATCH_DECISION = "dispatch_decision"
    HOLD = "hold"
    RELEASE = "release"
    SIGNAL_CHANGE = "signal_change"
    MAINTENANCE_START = "maintenance_start"
    MAINTENANCE_END = "maintenance_end"

@dataclass
class Event:
    event_type: EventType
    payload: Dict[str, Any]

@dataclass
class EventLogEntry:
    time: float
    event_type: EventType
    train_id: Optional[str]
    section_id: Optional[str]
    details: Dict[str, Any]

@dataclass
class Action:
    pass

@dataclass
class HoldAction(Action):
    train_id: str

@dataclass
class ReleaseAction(Action):
    train_id: str

@dataclass
class RouteAction(Action):
    train_id: str
    next_section: str

class Section:
    def __init__(self, id: str, length_m: float, max_speed_kmph: Optional[float] = None):
        self.id = id
        self.length_m = length_m
        self.max_speed_kmph = max_speed_kmph
        self.occupied_until = 0.0
        self.current_train: Optional[str] = None
        self.connected_switches: List[str] = []
    
    def is_free_at(self, t: float) -> bool:
        return t >= self.occupied_until
    
    def reserve(self, train_id: str, entry_time: float, exit_time: float):
        self.occupied_until = exit_time
        self.current_train = train_id

class Switch:
    def __init__(self, id: str):
        self.id = id
        self.locked_until = 0.0
        self.locked_by: Optional[str] = None
    
    def reserve(self, train_id: str, start_t: float, end_t: float):
        self.locked_until = end_t
        self.locked_by = train_id

class Platform:
    def __init__(self, id: str, capacity: int):
        self.id = id
        self.capacity = capacity
        self.occupied_untils: List[Tuple[str, float]] = []
    
    def available_at(self, t: float) -> bool:
        # Clean expired occupancies
        self.occupied_untils = [(train_id, until_t) for train_id, until_t in self.occupied_untils if until_t > t]
        return len(self.occupied_untils) < self.capacity
    
    def reserve(self, train_id: str, start_t: float, leave_t: float):
        self.occupied_untils.append((train_id, leave_t))

@dataclass
class EventTimestamp:
    section_id: str
    entry_time: float
    exit_time: Optional[float] = None

class Train:
    def __init__(self, id: str, type: str, priority: int, route: List[str], 
                 sched_entry_time: float, speed_kmph: float, dwell_profile: Dict[str, float] = None):
        self.id = id
        self.type = type
        self.priority = priority
        self.route = route
        self.sched_entry_time = sched_entry_time
        self.speed_kmph = speed_kmph
        self.dwell_profile = dwell_profile or {}
        self.route_index = 0
        self.state = "waiting"  # waiting, moving, at_platform
        self.timeline: List[EventTimestamp] = []
        self.held = False
    
    def next_section(self) -> Optional[str]:
        if self.route_index < len(self.route):
            return self.route[self.route_index]
        return None
    
    def travel_time(self, section: Section) -> float:
        speed = min(self.speed_kmph, section.max_speed_kmph or float('inf'))
        return section.length_m / (speed * 1000 / 3600)  # Convert kmph to m/s

class Controller(ABC):
    @abstractmethod
    def decide(self, state: Dict[str, Any], mask: List[bool]) -> List[Action]:
        pass

class SimpleController(Controller):
    """Simple FIFO controller that releases trains in order of arrival"""
    
    def decide(self, state: Dict[str, Any], mask: List[bool]) -> List[Action]:
        actions = []
        
        # Get held trains sorted by scheduled entry time
        held_trains = [(train['sched_entry_time'], train_id, train) 
                      for train_id, train in state['trains'].items() 
                      if train['held']]
        held_trains.sort()
        
        # Try to release earliest trains first
        for _, train_id, train in held_trains:
            next_sec = train.get('next_section')
            if next_sec and state['sections'][next_sec]['is_free']:
                actions.append(ReleaseAction(train_id))
        
        return actions

class Simulator:
    def __init__(self, decision_interval: float = 5.0, controller: Controller = None):
        self.time = 0.0
        self.event_queue = []
        self.sections: Dict[str, Section] = {}
        self.switches: Dict[str, Switch] = {}
        self.platforms: Dict[str, Platform] = {}
        self.trains: Dict[str, Train] = {}
        self.logs: List[EventLogEntry] = []
        self.decision_interval = decision_interval
        self.controller = controller or SimpleController()
        self.seq_counter = 0
    
    def load_scenario(self, json_obj: Dict[str, Any]):
        """Load scenario from JSON configuration"""
        # Load sections
        for sec_data in json_obj.get('sections', []):
            section = Section(
                sec_data['id'],
                sec_data['length_m'],
                sec_data.get('max_speed_kmph')
            )
            section.connected_switches = sec_data.get('connected_switches', [])
            self.sections[section.id] = section
        
        # Load switches
        for sw_data in json_obj.get('switches', []):
            switch = Switch(sw_data['id'])
            self.switches[switch.id] = switch
        
        # Load platforms
        for plat_data in json_obj.get('platforms', []):
            platform = Platform(plat_data['id'], plat_data['capacity'])
            self.platforms[platform.id] = platform
        
        # Load trains
        for train_data in json_obj.get('trains', []):
            train = Train(
                train_data['id'],
                train_data['type'],
                train_data['priority'],
                train_data['route'],
                train_data['sched_entry_time'],
                train_data['speed_kmph'],
                train_data.get('dwell_profile', {})
            )
            self.trains[train.id] = train
            
            # Schedule initial train arrival
            self.schedule_event(
                train.sched_entry_time,
                EventType.TRAIN_ARRIVAL_AT_SECTION,
                {'train_id': train.id, 'section_id': train.route[0]}
            )
        
        # Schedule periodic controller decisions
        self.schedule_event(
            self.decision_interval,
            EventType.DISPATCH_DECISION,
            {}
        )
    
    def schedule_event(self, time: float, event_type: EventType, payload: Dict[str, Any]):
        """Add event to priority queue"""
        event = Event(event_type, payload)
        heapq.heappush(self.event_queue, (time, self.seq_counter, event))
        self.seq_counter += 1
    
    def run(self, until_time: Optional[float] = None):
        """Main simulation loop"""
        while self.event_queue:
            event_time, _, event = heapq.heappop(self.event_queue)
            
            if until_time and event_time > until_time:
                break
            
            self.time = event_time
            self._handle_event(event)
    
    def _handle_event(self, event: Event):
        """Process individual events"""
        payload = event.payload
        
        if event.event_type == EventType.TRAIN_ARRIVAL_AT_SECTION:
            self._handle_train_arrival(payload['train_id'], payload['section_id'])
        
        elif event.event_type == EventType.TRAIN_LEAVES_SECTION:
            self._handle_train_departure(payload['train_id'], payload['section_id'])
        
        elif event.event_type == EventType.DISPATCH_DECISION:
            self._handle_controller_decision()
            # Schedule next decision
            self.schedule_event(
                self.time + self.decision_interval,
                EventType.DISPATCH_DECISION,
                {}
            )
        
        # Log the event
        self.log_event(event.event_type, payload.get('train_id'), 
                      payload.get('section_id'), payload)
    
    def _handle_train_arrival(self, train_id: str, section_id: str):
        """Handle train requesting to enter a section"""
        train = self.trains[train_id]
        section = self.sections[section_id]
        
        # Check if section is available
        if not section.is_free_at(self.time):
            # Hold train until section is free
            train.held = True
            train.state = "waiting"
            return
        
        # Check switch conflicts
        for switch_id in section.connected_switches:
            switch = self.switches[switch_id]
            if switch.locked_until > self.time:
                train.held = True
                train.state = "waiting"
                return
        
        # Reserve section and switches
        travel_time = train.travel_time(section)
        exit_time = self.time + travel_time
        
        section.reserve(train_id, self.time, exit_time)
        
        for switch_id in section.connected_switches:
            self.switches[switch_id].reserve(train_id, self.time, exit_time)
        
        # Update train state
        train.state = "moving"
        train.timeline.append(EventTimestamp(section_id, self.time))
        
        # Schedule departure
        self.schedule_event(
            exit_time,
            EventType.TRAIN_LEAVES_SECTION,
            {'train_id': train_id, 'section_id': section_id}
        )
    
    def _handle_train_departure(self, train_id: str, section_id: str):
        """Handle train leaving a section"""
        train = self.trains[train_id]
        section = self.sections[section_id]
        
        # Free the section
        section.current_train = None
        
        # Update timeline
        if train.timeline and train.timeline[-1].section_id == section_id:
            train.timeline[-1].exit_time = self.time
        
        # Move to next section in route
        train.route_index += 1
        
        if train.route_index < len(train.route):
            next_section_id = train.route[train.route_index]
            
            # Schedule arrival at next section
            self.schedule_event(
                self.time + 0.1,  # Small delay for signal processing
                EventType.TRAIN_ARRIVAL_AT_SECTION,
                {'train_id': train_id, 'section_id': next_section_id}
            )
        else:
            # Train completed route
            train.state = "completed"
    
    def _handle_controller_decision(self):
        """Let controller make decisions about held trains"""
        state = self.get_state()
        mask = self._get_action_mask()
        
        actions = self.controller.decide(state, mask)
        
        for action in actions:
            self.apply_controller_action(action)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current simulation state for controller"""
        return {
            'time': self.time,
            'trains': {tid: {
                'id': t.id,
                'state': t.state,
                'held': t.held,
                'route_index': t.route_index,
                'next_section': t.next_section(),
                'sched_entry_time': t.sched_entry_time
            } for tid, t in self.trains.items()},
            'sections': {sid: {
                'id': s.id,
                'is_free': s.is_free_at(self.time),
                'current_train': s.current_train
            } for sid, s in self.sections.items()}
        }
    
    def _get_action_mask(self) -> List[bool]:
        """Get mask of valid actions"""
        # Simplified - all actions valid for now
        return [True] * len(self.trains)
    
    def apply_controller_action(self, action: Action):
        """Apply controller action"""
        if isinstance(action, ReleaseAction):
            train = self.trains[action.train_id]
            if train.held:
                train.held = False
                # Retry entry to current section
                next_sec = train.next_section()
                if next_sec:
                    self.schedule_event(
                        self.time + 0.1,
                        EventType.TRAIN_ARRIVAL_AT_SECTION,
                        {'train_id': train.id, 'section_id': next_sec}
                    )
    
    def log_event(self, event_type: EventType, train_id: Optional[str], 
                  section_id: Optional[str], details: Dict[str, Any]):
        """Log simulation event"""
        entry = EventLogEntry(self.time, event_type, train_id, section_id, details)
        self.logs.append(entry)
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute simulation metrics"""
        completed_trains = [t for t in self.trains.values() if t.state == "completed"]
        
        total_delay = 0
        for train in completed_trains:
            if train.timeline:
                actual_completion = train.timeline[-1].exit_time or self.time
                # Simple delay calculation (would need more sophisticated scheduling)
                scheduled_completion = train.sched_entry_time + 100  # Simplified
                delay = max(0, actual_completion - scheduled_completion)
                total_delay += delay
        
        return {
            'total_trains': len(self.trains),
            'completed_trains': len(completed_trains),
            'completion_rate': len(completed_trains) / len(self.trains) if self.trains else 0,
            'average_delay': total_delay / len(completed_trains) if completed_trains else 0,
            'total_simulation_time': self.time
        }
        import os

    def save_logs(self, filepath="logs/run1.json"):
        """Save all simulation logs to a JSON file for later visualization."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Create 'logs/' folder if missing

        logs_as_dict = []
        for log in self.logs:
            logs_as_dict.append({
                "time": log.time,
                "event_type": log.event_type.value,
                "train_id": log.train_id,
                "section_id": log.section_id,
                "details": log.details
            })

        with open(filepath, "w") as f:
            json.dump(logs_as_dict, f, indent=4)

        print(f"\nLogs successfully saved to {filepath}")

    def reset(self):
        """Reset simulation state"""
        self.time = 0.0
        self.event_queue.clear()
        self.logs.clear()
        self.seq_counter = 0
        
        # Reset all entities
        for section in self.sections.values():
            section.occupied_until = 0.0
            section.current_train = None
        
        for switch in self.switches.values():
            switch.locked_until = 0.0
            switch.locked_by = None
        
        for platform in self.platforms.values():
            platform.occupied_untils.clear()
        
        for train in self.trains.values():
            train.route_index = 0
            train.state = "waiting"
            train.timeline.clear()
            train.held = False

# Example usage
if __name__ == "__main__":
    # Example scenario
    scenario = {
        "sections": [
            {"id": "S1", "length_m": 1000, "max_speed_kmph": 80, "connected_switches": ["SW1"]},
            {"id": "S2", "length_m": 1500, "max_speed_kmph": 100, "connected_switches": ["SW1"]},
            {"id": "S3", "length_m": 800, "max_speed_kmph": 60}
        ],
        "switches": [
            {"id": "SW1"}
        ],
        "platforms": [
            {"id": "P1", "capacity": 2}
        ],
        "trains": [
            {
                "id": "T1",
                "type": "express",
                "priority": 1,
                "route": ["S1", "S2", "S3"],
                "sched_entry_time": 0.0,
                "speed_kmph": 80
            },
            {
                "id": "T2",
                "type": "local",
                "priority": 2,
                "route": ["S1", "S2"],
                "sched_entry_time": 10.0,
                "speed_kmph": 60
            }
        ]
    }
    
    # Create and run simulation
    sim = Simulator(decision_interval=5.0)
    sim.load_scenario(scenario)
    sim.run(until_time=300.0)
    
    # Save logs for visualization
    sim.save_logs("logs/run1.json")

    # Print results
    metrics = sim.compute_metrics()
    print("Simulation Results:")
    print(f"Completed trains: {metrics['completed_trains']}/{metrics['total_trains']}")
    print(f"Completion rate: {metrics['completion_rate']:.2%}")
    print(f"Average delay: {metrics['average_delay']:.2f} seconds")
    print(f"Total simulation time: {metrics['total_simulation_time']:.2f} seconds")
    
    print("\nEvent Log (last 10 events):")
    for log in sim.logs[-10:]:
        if log.train_id is not None and log.section_id is not None:
            print(f"{log.time:.2f}s: {log.event_type.value} - Train: {log.train_id}, Section: {log.section_id}")
