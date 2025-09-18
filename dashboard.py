#!/usr/bin/env python3
"""
Train Traffic Management Dashboard (Refactored)
A Pygame-based visualization for railway network state and train management.
"""

import pygame
import json
import os
from typing import Dict, List, Any, Tuple

# --- Constants ---
# Colors
BLACK = (20, 20, 30)
WHITE = (230, 230, 230)
GREEN = (40, 167, 69)
RED = (220, 53, 69)
BLUE = (0, 123, 255)
DARK_GRAY = (52, 58, 64)
LIGHT_GRAY = (150, 150, 150)
YELLOW = (255, 193, 7)

# Display
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
SECTION_WIDTH = 150
SECTION_HEIGHT = 60
BUTTON_WIDTH = 100
BUTTON_HEIGHT = 30
SECTION_SPACING = 20

class TrainDashboard:
    """A Pygame dashboard to visualize and manage train traffic."""
    
    def __init__(self, scenario_path="scenarios/scenario2.json", simulator_path="logs/run1.json", optimizer_path="optimizer_schedule.json", simulator_state_path="Optimizer/simulator_state.json"):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Train Traffic Management Dashboard")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 30)
        self.small_font = pygame.font.Font(None, 22)
        
        self.scenario_path = scenario_path
        self.simulator_path = simulator_path
        self.optimizer_path = optimizer_path
        self.simulator_state_path = simulator_state_path
        
        self.sections: Dict[str, Any] = {}
        self.trains: Dict[str, Any] = {}
        self.schedule: List[Dict[str, Any]] = []
        self.optimizer_metadata: Dict[str, Any] = {}
        
        # Store button rectangles for click detection
        self.buttons: Dict[str, pygame.Rect] = {}
        self.running = True

    def load_and_parse_data(self):
        """Loads all data sources and computes the final state of the simulation."""
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # 1. Try to load from simulator state first (most current data)
        try:
            state_file = os.path.join(script_dir, self.simulator_state_path)
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            
            # Load blocks as sections
            self.sections = {}
            for block_id, block_info in state_data.get('blocks', {}).items():
                self.sections[block_id] = {
                    'id': block_id,
                    'is_free': block_info.get('is_free', True),
                    'current_train': block_info.get('current_train'),
                    'length_m': 1000,  # Default length
                    'max_speed_kmph': 80  # Default speed
                }
            
            # Load trains
            self.trains = {}
            for train_id, train_info in state_data.get('trains', {}).items():
                self.trains[train_id] = {
                    'id': train_id,
                    'type': 'express' if train_info.get('priority', 1) == 1 else 'local',
                    'priority': train_info.get('priority', 1),
                    'state': train_info.get('state', 'waiting'),
                    'current_section': train_info.get('current_block'),
                    'route': train_info.get('route', []),
                    'delay': train_info.get('current_delay', 0)
                }
            
            print(f"Successfully loaded simulator state from {self.simulator_state_path}")
        except Exception as e:
            print(f"Warning: Could not load simulator state: {e}")
            # Fallback to scenario + event log processing
            self._load_from_scenario_and_events(script_dir)

        # 2. Load the optimizer schedule
        try:
            opt_file = os.path.join(script_dir, self.optimizer_path)
            with open(opt_file, 'r') as f:
                opt_data = json.load(f)
            
            # Extract train actions as schedule
            self.schedule = opt_data.get('train_actions', [])
            self.optimizer_metadata = opt_data.get('optimization_metadata', {})
            print(f"Successfully loaded optimizer schedule from {self.optimizer_path}")
        except Exception as e:
            print(f"Warning: Optimizer schedule not found: {e}")
            self.schedule = []
            self.optimizer_metadata = {}

    def _load_from_scenario_and_events(self, script_dir):
        """Fallback method to load from scenario and event log."""
        try:
            scenario_file = os.path.join(script_dir, self.scenario_path)
            with open(scenario_file, 'r') as f:
                scenario = json.load(f)
            
            # Initialize sections as all free
            for section_data in scenario.get('sections', []):
                self.sections[section_data['id']] = {**section_data, 'is_free': True, 'current_train': None}
            
            # Initialize trains
            for train_data in scenario.get('trains', []):
                self.trains[train_data['id']] = {**train_data, 'state': 'waiting', 'current_section': None}
            
            # Process events
            sim_file = os.path.join(script_dir, self.simulator_path)
            with open(sim_file, 'r') as f:
                events = json.load(f)
            
            events.sort(key=lambda x: x.get('time', 0))
            
            for event in events:
                event_type = event.get('event_type')
                train_id = event.get('train_id')
                section_id = event.get('section_id')

                if event_type == 'train_arrival_at_section':
                    if section_id in self.sections:
                        self.sections[section_id].update({'is_free': False, 'current_train': train_id})
                    if train_id in self.trains:
                        self.trains[train_id].update({'state': 'moving', 'current_section': section_id})
                
                elif event_type == 'train_leaves_section':
                    if section_id in self.sections and self.sections[section_id]['current_train'] == train_id:
                        self.sections[section_id].update({'is_free': True, 'current_train': None})
                    if train_id in self.trains:
                        self.trains[train_id].update({'state': 'waiting'})
            
            print(f"Successfully loaded from scenario and events")
        except Exception as e:
            print(f"Error in fallback loading: {e}")

    def draw_track_sections(self):
        """Draws the main track visualization."""
        start_x = 50
        start_y = 80
        for i, (section_id, data) in enumerate(self.sections.items()):
            x = start_x + i * (SECTION_WIDTH + SECTION_SPACING)
            rect = pygame.Rect(x, start_y, SECTION_WIDTH, SECTION_HEIGHT)
            
            color = GREEN if data['is_free'] else RED
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
            pygame.draw.rect(self.screen, WHITE, rect, 2, border_radius=5)
            
            # Draw Section ID
            id_text = self.font.render(section_id, True, WHITE)
            self.screen.blit(id_text, id_text.get_rect(center=(rect.centerx, rect.centery - 10)))
            
            # Draw Train ID if present
            if data['current_train']:
                train_text = self.small_font.render(data['current_train'], True, BLACK)
                self.screen.blit(train_text, train_text.get_rect(center=(rect.centerx, rect.centery + 15)))

    def draw_train_states(self):
        """Draws the train information panel."""
        panel_rect = pygame.Rect(50, 160, WINDOW_WIDTH - 100, 120)
        pygame.draw.rect(self.screen, DARK_GRAY, panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, panel_rect, 2, border_radius=10)

        title_text = self.font.render("Train States", True, WHITE)
        self.screen.blit(title_text, (panel_rect.x + 20, panel_rect.y + 15))
        
        # Draw train information in two columns
        start_x = panel_rect.x + 20
        start_y = panel_rect.y + 50
        col_width = (panel_rect.width - 40) // 2
        
        for i, (train_id, train_data) in enumerate(self.trains.items()):
            col = i % 2
            row = i // 2
            x = start_x + col * col_width
            y = start_y + row * 25
            
            # Train info
            state_icon = "🚂" if train_data['state'] == 'moving' else "⏸️" if train_data['state'] == 'waiting' else "✅"
            current_section = train_data.get('current_section', 'None')
            delay = train_data.get('delay', 0)
            
            train_info = f"{state_icon} {train_id} ({train_data['type']}) - {train_data['state']}"
            if current_section != 'None':
                train_info += f" - {current_section}"
            if delay > 0:
                train_info += f" (Delay: {delay:.1f}s)"
            
            train_text = self.small_font.render(train_info, True, WHITE)
            self.screen.blit(train_text, (x, y))

    def draw_schedule_panel(self):
        """Draws the interactive schedule management panel."""
        panel_rect = pygame.Rect(50, 300, WINDOW_WIDTH - 100, WINDOW_HEIGHT - 350)
        pygame.draw.rect(self.screen, DARK_GRAY, panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, panel_rect, 2, border_radius=10)

        title_text = self.font.render("Optimizer Schedule", True, WHITE)
        self.screen.blit(title_text, (panel_rect.x + 20, panel_rect.y + 20))
        
        # Show metadata if available
        if self.optimizer_metadata:
            meta_text = f"Congestion: {self.optimizer_metadata.get('congestion_level', 0):.1%} | "
            meta_text += f"Waiting Trains: {self.optimizer_metadata.get('waiting_trains', 0)} | "
            meta_text += f"Total Actions: {self.optimizer_metadata.get('total_actions', 0)}"
            meta_surface = self.small_font.render(meta_text, True, LIGHT_GRAY)
            self.screen.blit(meta_surface, (panel_rect.x + 20, panel_rect.y + 50))
        
        self.buttons.clear() # Clear old buttons before redrawing
        if not self.schedule:
            no_data_text = self.small_font.render("Waiting for optimizer_schedule.json...", True, YELLOW)
            self.screen.blit(no_data_text, (panel_rect.x + 20, panel_rect.y + 80))
            return

        start_y = panel_rect.y + 90
        for i, item in enumerate(self.schedule):
            y = start_y + i * 50
            # Draw schedule item text
            action_type = item.get('action_type', 'unknown')
            target_block = item.get('target_block', 'N/A')
            duration = item.get('estimated_duration', 0)
            
            info = f"Train: {item['train_id']} | Action: {action_type} | Target: {target_block} | Duration: {duration:.1f}s"
            info_text = self.small_font.render(info, True, WHITE)
            self.screen.blit(info_text, (panel_rect.x + 20, y))
            
            # Draw buttons and store their Rects for click detection
            # Accept Button
            accept_rect = pygame.Rect(panel_rect.right - 2 * (BUTTON_WIDTH + 10), y - 5, BUTTON_WIDTH, BUTTON_HEIGHT)
            pygame.draw.rect(self.screen, GREEN, accept_rect, border_radius=5)
            accept_text = self.small_font.render("Accept", True, BLACK)
            self.screen.blit(accept_text, accept_text.get_rect(center=accept_rect.center))
            self.buttons[f"accept_{i}"] = accept_rect
            
            # Override Button
            override_rect = pygame.Rect(panel_rect.right - (BUTTON_WIDTH + 10), y - 5, BUTTON_WIDTH, BUTTON_HEIGHT)
            pygame.draw.rect(self.screen, YELLOW, override_rect, border_radius=5)
            override_text = self.small_font.render("Override", True, BLACK)
            self.screen.blit(override_text, override_text.get_rect(center=override_rect.center))
            self.buttons[f"override_{i}"] = override_rect

    def handle_mouse_click(self, pos: Tuple[int, int]):
        """Handles clicks on interactive UI elements."""
        for name, rect in self.buttons.items():
            if rect.collidepoint(pos):
                action_type, index_str = name.split('_')
                index = int(index_str)
                schedule_item = self.schedule[index]
                train_id = schedule_item['train_id']
                
                if action_type == 'accept':
                    print(f"Action ACCEPTED for train {train_id}: {schedule_item.get('action_type', 'unknown')}")
                elif action_type == 'override':
                    # Determine the opposite action for the feedback
                    current_action = schedule_item.get('action_type', 'move_to_block')
                    if current_action == 'move_to_block':
                        overridden_action = 'hold'
                    else:
                        overridden_action = 'move_to_block'
                    
                    feedback = {"user_feedback": {"train_id": train_id, "overridden_action": overridden_action}}
                    print("--- OVERRIDE ---")
                    print(json.dumps(feedback, indent=2))
                    print("----------------")
                return # Stop after one button is found

    def run(self):
        """Main application loop."""
        print("Starting Dashboard... Press 'R' to reload data, 'ESC' to quit.")
        self.load_and_parse_data() # Initial load
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_r:
                        print("\nReloading data...")
                        self.load_and_parse_data()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: # Left click
                        self.handle_mouse_click(event.pos)
            
            self.screen.fill(BLACK)
            self.draw_track_sections()
            self.draw_train_states()
            self.draw_schedule_panel()
            pygame.display.flip()
            self.clock.tick(30) # Run at 30 FPS

        pygame.quit()
        print("Dashboard closed.")

def main():
    """Main entry point"""
    dashboard = TrainDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()