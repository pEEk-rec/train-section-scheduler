import json
import math
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import itertools
from abc import ABC, abstractmethod

class ActionType(Enum):
    MOVE_TO_BLOCK = "move_to_block"
    HOLD = "hold"
    DISPATCH = "dispatch"
    PLATFORM_ENTRY = "platform_entry"
    PLATFORM_EXIT = "platform_exit"

@dataclass
class TrainAction:
    """Represents a single train action"""
    train_id: str
    action_type: ActionType
    target_block: Optional[str] = None
    target_platform: Optional[str] = None
    estimated_duration: float = 0.0
    priority_override: bool = False

@dataclass
class OptimizationWeights:
    """Weights for multi-objective optimization"""
    priority_weight: float = 1.0
    delay_reduction_weight: float = 0.8
    throughput_weight: float = 0.6
    congestion_penalty_weight: float = 0.4
    rl_prediction_weight: float = 0.5

@dataclass
class SystemConstraints:
    """System-wide operational constraints"""
    max_moves_per_interval: int = 5
    min_block_separation: int = 1  # minimum blocks between trains
    max_platform_occupancy: float = 0.9  # 90% of platform capacity
    priority_override_threshold: float = 60.0  # seconds of delay before priority override

@dataclass
class TrainMetrics:
    """Computed metrics for a train"""
    train_id: str
    priority: int
    current_delay: float
    scheduled_delay: float
    blocks_to_destination: int
    estimated_travel_time: float
    congestion_impact: float
    rl_predicted_benefit: float
    feasible_moves: List[str]  # list of block IDs train can move to

class SafetyValidator:
    """Validates that proposed actions meet safety constraints"""
    
    @staticmethod
    def validate_block_occupancy(current_state: Dict[str, Any], 
                               proposed_actions: List[TrainAction]) -> bool:
        """Ensure only one train per block after actions"""
        # Track block assignments after actions
        block_assignments = {}
        
        # Current occupancy
        for block_id, block_info in current_state['blocks'].items():
            if not block_info['is_free']:
                block_assignments[block_id] = block_info['current_train']
        
        # Apply proposed actions
        for action in proposed_actions:
            if action.action_type == ActionType.MOVE_TO_BLOCK:
                if action.target_block in block_assignments:
                    return False  # Block would be double-occupied
                block_assignments[action.target_block] = action.train_id
                
                # Remove train from current block
                for block_id, train_id in list(block_assignments.items()):
                    if train_id == action.train_id and block_id != action.target_block:
                        del block_assignments[block_id]
        
        return True
    
    @staticmethod
    def validate_switch_conflicts(current_state: Dict[str, Any],
                                proposed_actions: List[TrainAction]) -> bool:
        """Ensure no switch conflicts"""
        switches = current_state.get('switches', {})
        
        # Check if any switches are locked by actions
        switch_usage = {}
        
        for action in proposed_actions:
            if action.action_type == ActionType.MOVE_TO_BLOCK:
                # Simplified: assume each block movement might use a switch
                # In reality, you'd check routing tables
                for switch_id, switch_info in switches.items():
                    if not switch_info.get('is_available', True):
                        continue
                    
                    # Check if this movement conflicts with switch
                    # This is simplified - real implementation would check routing
                    if action.target_block in ['B3', 'B4']:  # Example conflict blocks
                        if switch_id in switch_usage:
                            return False  # Conflict
                        switch_usage[switch_id] = action.train_id
        
        return True
    
    @staticmethod
    def validate_platform_capacity(current_state: Dict[str, Any],
                                 proposed_actions: List[TrainAction]) -> bool:
        """Ensure platform capacity not exceeded"""
        platforms = current_state.get('platforms', {})
        
        for action in proposed_actions:
            if action.action_type == ActionType.PLATFORM_ENTRY:
                platform = platforms.get(action.target_platform, {})
                current_occupancy = platform.get('current_occupancy', 0)
                capacity = platform.get('capacity', 1)
                
                if current_occupancy >= capacity:
                    return False
        
        return True
    
    @staticmethod
    def validate_minimum_separation(current_state: Dict[str, Any],
                                  proposed_actions: List[TrainAction],
                                  min_separation: int = 1) -> bool:
        """Ensure minimum separation between trains"""
        # Simplified implementation - would need block topology
        # For now, just ensure no adjacent blocks are occupied by same actions
        occupied_blocks = set()
        
        for action in proposed_actions:
            if action.action_type == ActionType.MOVE_TO_BLOCK:
                occupied_blocks.add(action.target_block)
        
        # Check if any blocks are too close (simplified)
        for block in occupied_blocks:
            block_num = int(block[1:]) if block[1:].isdigit() else 0
            adjacent_blocks = [f"B{block_num-1}", f"B{block_num+1}"]
            
            for adj_block in adjacent_blocks:
                if adj_block in occupied_blocks and adj_block != block:
                    return False  # Violation of minimum separation
        
        return True

class RLAgentInterface(ABC):
    """Interface for RL agent predictions"""
    
    @abstractmethod
    def get_predictions(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get RL agent predictions for current state"""
        pass

class MockRLAgent(RLAgentInterface):
    """Mock RL agent for testing"""
    
    def get_predictions(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock predictions"""
        predictions = {}
        
        for train_id, train_info in current_state.get('trains', {}).items():
            # Mock prediction based on train priority and delay
            priority = train_info.get('priority', 5)
            delay = train_info.get('current_delay', 0)
            
            predicted_benefit = max(0, (10 - priority) * 2 + delay * 0.1)
            next_block = train_info.get('next_block')
            
            predictions[train_id] = {
                'predicted_next_block': next_block,
                'expected_delay_reduction': predicted_benefit,
                'confidence': 0.8
            }
        
        return predictions

class BlockSchedulerOptimizer:
    """Main optimizer for block-based train scheduling"""
    
    def __init__(self, 
                 weights: OptimizationWeights = None,
                 constraints: SystemConstraints = None,
                 rl_agent: RLAgentInterface = None):
        self.weights = weights or OptimizationWeights()
        self.constraints = constraints or SystemConstraints()
        self.rl_agent = rl_agent or MockRLAgent()
        self.safety_validator = SafetyValidator()
        
        # Metrics tracking
        self.optimization_history = []
        self.performance_metrics = {
            'total_optimizations': 0,
            'successful_moves': 0,
            'safety_violations_prevented': 0,
            'average_optimization_time': 0.0
        }
    
    def optimize(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Main optimization function"""
        import time
        start_time = time.time()
        
        # Step 1: Get RL agent predictions
        rl_predictions = self.rl_agent.get_predictions(current_state)
        
        # Step 2: Analyze system state
        system_analysis = self._analyze_system_state(current_state)
        
        # Step 3: Compute train metrics
        train_metrics = self._compute_train_metrics(current_state, rl_predictions, system_analysis)
        
        # Step 4: Generate candidate actions
        candidate_actions = self._generate_candidate_actions(current_state, train_metrics)
        
        # Step 5: Evaluate and select optimal actions
        optimal_actions = self._select_optimal_actions(candidate_actions, train_metrics, system_analysis)
        
        # Step 6: Validate safety
        if not self._validate_all_constraints(current_state, optimal_actions):
            # Fallback to safer subset of actions
            optimal_actions = self._generate_safe_fallback(current_state, train_metrics)
        
        # Step 7: Format output
        output = self._format_output(optimal_actions, system_analysis, rl_predictions)
        
        # Update metrics
        optimization_time = time.time() - start_time
        self._update_performance_metrics(optimal_actions, optimization_time)
        
        return output
    
    def _analyze_system_state(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current system state for decision making"""
        blocks = current_state.get('blocks', {})
        trains = current_state.get('trains', {})
        
        # Calculate congestion metrics
        total_blocks = len(blocks)
        occupied_blocks = sum(1 for b in blocks.values() if not b['is_free'])
        congestion_level = occupied_blocks / max(1, total_blocks)
        
        # Identify bottlenecks
        bottleneck_blocks = []
        for block_id, block_info in blocks.items():
            if not block_info['is_free']:
                # Check if trains are queued for this block
                queued_trains = sum(1 for t in trains.values() 
                                  if t.get('next_block') == block_id and t.get('state') == 'waiting')
                if queued_trains > 1:
                    bottleneck_blocks.append(block_id)
        
        # Calculate system metrics
        waiting_trains = [t for t in trains.values() if t.get('state') == 'waiting']
        moving_trains = [t for t in trains.values() if t.get('state') == 'moving']
        
        return {
            'total_blocks': total_blocks,
            'occupied_blocks': occupied_blocks,
            'free_blocks': total_blocks - occupied_blocks,
            'congestion_level': congestion_level,
            'bottleneck_blocks': bottleneck_blocks,
            'waiting_trains_count': len(waiting_trains),
            'moving_trains_count': len(moving_trains),
            'time': current_state.get('time', 0)
        }
    
    def _compute_train_metrics(self, current_state: Dict[str, Any], 
                              rl_predictions: Dict[str, Any],
                              system_analysis: Dict[str, Any]) -> List[TrainMetrics]:
        """Compute detailed metrics for each train"""
        trains = current_state.get('trains', {})
        blocks = current_state.get('blocks', {})
        current_time = current_state.get('time', 0)
        
        train_metrics = []
        
        for train_id, train_info in trains.items():
            # Basic train info
            priority = train_info.get('priority', 5)
            state = train_info.get('state', 'unknown')
            current_block = train_info.get('current_block')
            next_block = train_info.get('next_block')
            sched_entry_time = train_info.get('sched_entry_time', 0)
            
            # Calculate delays
            scheduled_delay = max(0, current_time - sched_entry_time)
            current_delay = train_info.get('current_delay', scheduled_delay)
            
            # Estimate travel metrics
            blocks_to_destination = self._estimate_blocks_to_destination(train_info)
            estimated_travel_time = blocks_to_destination * 30.0  # 30 seconds per block (simplified)
            
            # Calculate congestion impact
            congestion_impact = self._calculate_congestion_impact(train_info, system_analysis)
            
            # Get RL prediction benefit
            rl_benefit = 0.0
            if train_id in rl_predictions:
                rl_benefit = rl_predictions[train_id].get('expected_delay_reduction', 0.0)
            
            # Find feasible moves
            feasible_moves = self._find_feasible_moves(train_info, blocks)
            
            metrics = TrainMetrics(
                train_id=train_id,
                priority=priority,
                current_delay=current_delay,
                scheduled_delay=scheduled_delay,
                blocks_to_destination=blocks_to_destination,
                estimated_travel_time=estimated_travel_time,
                congestion_impact=congestion_impact,
                rl_predicted_benefit=rl_benefit,
                feasible_moves=feasible_moves
            )
            
            train_metrics.append(metrics)
        
        return train_metrics
    
    def _estimate_blocks_to_destination(self, train_info: Dict[str, Any]) -> int:
        """Estimate remaining blocks to destination"""
        # Simplified - in reality would use route planning
        route = train_info.get('route', [])
        current_index = train_info.get('route_index', 0)
        return max(0, len(route) - current_index)
    
    def _calculate_congestion_impact(self, train_info: Dict[str, Any], 
                                   system_analysis: Dict[str, Any]) -> float:
        """Calculate how much this train contributes to congestion"""
        priority = train_info.get('priority', 5)
        congestion_level = system_analysis['congestion_level']
        
        # Higher priority trains have more impact when system is congested
        base_impact = (10 - priority) * 0.1
        congestion_multiplier = 1.0 + congestion_level
        
        return base_impact * congestion_multiplier
    
    def _find_feasible_moves(self, train_info: Dict[str, Any], 
                           blocks: Dict[str, Any]) -> List[str]:
        """Find blocks this train can legally move to"""
        feasible_moves = []
        next_block = train_info.get('next_block')
        
        if next_block and blocks.get(next_block, {}).get('is_free', False):
            feasible_moves.append(next_block)
        
        # Could add alternative routing options here
        return feasible_moves
    
    def _generate_candidate_actions(self, current_state: Dict[str, Any],
                                  train_metrics: List[TrainMetrics]) -> List[List[TrainAction]]:
        """Generate all possible combinations of actions"""
        candidate_combinations = []
        
        # Generate individual actions for each train
        individual_actions = {}
        for metrics in train_metrics:
            train_actions = []
            
            # Add move actions
            for block_id in metrics.feasible_moves:
                action = TrainAction(
                    train_id=metrics.train_id,
                    action_type=ActionType.MOVE_TO_BLOCK,
                    target_block=block_id,
                    estimated_duration=30.0  # Simplified
                )
                train_actions.append(action)
            
            # Add hold action
            hold_action = TrainAction(
                train_id=metrics.train_id,
                action_type=ActionType.HOLD
            )
            train_actions.append(hold_action)
            
            individual_actions[metrics.train_id] = train_actions
        
        # Generate combinations (limited to avoid explosion)
        train_ids = list(individual_actions.keys())[:self.constraints.max_moves_per_interval]
        
        # Create all combinations of actions
        action_combinations = []
        if train_ids:
            for combination in itertools.product(*[individual_actions[tid] for tid in train_ids]):
                # Filter out combinations with only hold actions
                move_actions = [a for a in combination if a.action_type != ActionType.HOLD]
                if move_actions:
                    action_combinations.append(list(combination))
        
        return action_combinations[:50]  # Limit combinations for performance
    
    def _select_optimal_actions(self, candidate_actions: List[List[TrainAction]],
                              train_metrics: List[TrainMetrics],
                              system_analysis: Dict[str, Any]) -> List[TrainAction]:
        """Select the optimal set of actions"""
        if not candidate_actions:
            return []
        
        best_actions = []
        best_score = float('-inf')
        
        # Create metrics lookup
        metrics_lookup = {tm.train_id: tm for tm in train_metrics}
        
        for action_set in candidate_actions:
            score = self._calculate_objective_score(action_set, metrics_lookup, system_analysis)
            
            if score > best_score:
                best_score = score
                best_actions = action_set
        
        return best_actions
    
    def _calculate_objective_score(self, actions: List[TrainAction],
                                 metrics_lookup: Dict[str, TrainMetrics],
                                 system_analysis: Dict[str, Any]) -> float:
        """Calculate multi-objective score for action set"""
        total_score = 0.0
        
        for action in actions:
            if action.action_type == ActionType.HOLD:
                continue
            
            metrics = metrics_lookup.get(action.train_id)
            if not metrics:
                continue
            
            # Priority score (higher priority = higher score)
            priority_score = (10 - metrics.priority) * self.weights.priority_weight
            
            # Delay reduction score
            delay_reduction = min(metrics.current_delay, 60.0)  # Cap at 60 seconds
            delay_score = delay_reduction * self.weights.delay_reduction_weight
            
            # Throughput score (moving trains increases throughput)
            throughput_score = 10.0 * self.weights.throughput_weight
            
            # Congestion penalty
            congestion_penalty = (metrics.congestion_impact * 
                                system_analysis['congestion_level'] * 
                                self.weights.congestion_penalty_weight)
            
            # RL prediction benefit
            rl_score = metrics.rl_predicted_benefit * self.weights.rl_prediction_weight
            
            # Combine scores
            action_score = (priority_score + delay_score + throughput_score + 
                          rl_score - congestion_penalty)
            
            total_score += action_score
        
        return total_score
    
    def _validate_all_constraints(self, current_state: Dict[str, Any],
                                actions: List[TrainAction]) -> bool:
        """Validate all safety and operational constraints"""
        validators = [
            self.safety_validator.validate_block_occupancy,
            self.safety_validator.validate_switch_conflicts,
            self.safety_validator.validate_platform_capacity,
            lambda state, acts: self.safety_validator.validate_minimum_separation(
                state, acts, self.constraints.min_block_separation)
        ]
        
        for validator in validators:
            if not validator(current_state, actions):
                self.performance_metrics['safety_violations_prevented'] += 1
                return False
        
        return True
    
    def _generate_safe_fallback(self, current_state: Dict[str, Any],
                              train_metrics: List[TrainMetrics]) -> List[TrainAction]:
        """Generate safe fallback actions when optimal solution violates constraints"""
        safe_actions = []
        
        # Sort trains by priority and delay
        sorted_metrics = sorted(train_metrics, 
                              key=lambda tm: (tm.priority, -tm.current_delay))
        
        for metrics in sorted_metrics[:2]:  # Limit to 2 trains for safety
            if metrics.feasible_moves:
                action = TrainAction(
                    train_id=metrics.train_id,
                    action_type=ActionType.MOVE_TO_BLOCK,
                    target_block=metrics.feasible_moves[0]
                )
                
                # Validate single action
                if self._validate_all_constraints(current_state, [action]):
                    safe_actions.append(action)
        
        return safe_actions
    
    def _format_output(self, actions: List[TrainAction],
                      system_analysis: Dict[str, Any],
                      rl_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Format optimizer output as JSON"""
        output = {
            'timestamp': system_analysis['time'],
            'optimization_metadata': {
                'congestion_level': system_analysis['congestion_level'],
                'total_actions': len(actions),
                'waiting_trains': system_analysis['waiting_trains_count'],
                'bottleneck_blocks': system_analysis['bottleneck_blocks']
            },
            'train_actions': [],
            'rl_predictions_used': len(rl_predictions) > 0,
            'safety_validated': True
        }
        
        for action in actions:
            action_dict = {
                'train_id': action.train_id,
                'action_type': action.action_type.value,
                'estimated_duration': action.estimated_duration
            }
            
            if action.target_block:
                action_dict['target_block'] = action.target_block
            if action.target_platform:
                action_dict['target_platform'] = action.target_platform
            
            output['train_actions'].append(action_dict)
        
        return output
    
    def _update_performance_metrics(self, actions: List[TrainAction], 
                                  optimization_time: float):
        """Update performance tracking metrics"""
        self.performance_metrics['total_optimizations'] += 1
        move_actions = [a for a in actions if a.action_type != ActionType.HOLD]
        self.performance_metrics['successful_moves'] += len(move_actions)
        
        # Update average optimization time
        total_opts = self.performance_metrics['total_optimizations']
        current_avg = self.performance_metrics['average_optimization_time']
        new_avg = ((current_avg * (total_opts - 1)) + optimization_time) / total_opts
        self.performance_metrics['average_optimization_time'] = new_avg
    
    def save_decision_log(self, output: Dict[str, Any], filename: str):
        """Save decision output to JSON file"""
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance metrics report"""
        return {
            'performance_metrics': self.performance_metrics,
            'optimization_settings': {
                'weights': {
                    'priority': self.weights.priority_weight,
                    'delay_reduction': self.weights.delay_reduction_weight,
                    'throughput': self.weights.throughput_weight,
                    'congestion_penalty': self.weights.congestion_penalty_weight,
                    'rl_prediction': self.weights.rl_prediction_weight
                },
                'constraints': {
                    'max_moves_per_interval': self.constraints.max_moves_per_interval,
                    'min_block_separation': self.constraints.min_block_separation,
                    'max_platform_occupancy': self.constraints.max_platform_occupancy
                }
            }
        }

# Example usage and testing
if __name__ == "__main__":
    # Example system state
    example_state = {
        "time": 120.0,
        "blocks": {
            "B1": {"is_free": True, "current_train": None},
            "B2": {"is_free": False, "current_train": "T1"},
            "B3": {"is_free": True, "current_train": None},
            "B4": {"is_free": False, "current_train": "T2"},
            "B5": {"is_free": True, "current_train": None}
        },
        "switches": {
            "SW1": {"is_available": True, "current_position": "straight"},
            "SW2": {"is_available": True, "current_position": "diverging"}
        },
        "platforms": {
            "P1": {"capacity": 2, "current_occupancy": 1},
            "P2": {"capacity": 1, "current_occupancy": 0}
        },
        "trains": {
            "T1": {
                "priority": 1,
                "state": "moving",
                "current_block": "B2",
                "next_block": "B3",
                "sched_entry_time": 60.0,
                "current_delay": 15.0,
                "route": ["B1", "B2", "B3", "B4", "B5"],
                "route_index": 2
            },
            "T2": {
                "priority": 3,
                "state": "waiting",
                "current_block": "B4",
                "next_block": "B5",
                "sched_entry_time": 80.0,
                "current_delay": 25.0,
                "route": ["B3", "B4", "B5"],
                "route_index": 2
            },
            "T3": {
                "priority": 2,
                "state": "waiting",
                "current_block": None,
                "next_block": "B1",
                "sched_entry_time": 100.0,
                "current_delay": 20.0,
                "route": ["B1", "B2", "B3"],
                "route_index": 0
            }
        }
    }
    
    # Create optimizer with custom settings
    weights = OptimizationWeights(
        priority_weight=1.2,
        delay_reduction_weight=1.0,
        throughput_weight=0.8,
        congestion_penalty_weight=0.6,
        rl_prediction_weight=0.7
    )
    
    constraints = SystemConstraints(
        max_moves_per_interval=3,
        min_block_separation=1,
        max_platform_occupancy=0.8
    )
    
    optimizer = BlockSchedulerOptimizer(weights=weights, constraints=constraints)
    
    # Run optimization
    result = optimizer.optimize(example_state)
    
    # Print results
    print("=== OPTIMIZATION RESULT ===")
    print(json.dumps(result, indent=2))
    
    print(f"\n=== PERFORMANCE REPORT ===")
    performance = optimizer.get_performance_report()
    print(json.dumps(performance, indent=2))
    
    # Save to file
    optimizer.save_decision_log(result, "train_schedule_decision.json")
    print(f"\nDecision log saved to train_schedule_decision.json")