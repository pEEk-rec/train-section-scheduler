#!/usr/bin/env python3
"""
Train Traffic Management Dashboard (Animation and UI Fixed)
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
DARK_GRAY = (52, 58, 64)
LIGHT_GRAY = (150, 150, 150)
YELLOW = (255, 193, 7)
TRAIN_COLOR = (0, 191, 255)

# Display
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
SECTION_WIDTH = 150
SECTION_HEIGHT = 60
BUTTON_WIDTH = 100
BUTTON_HEIGHT = 30
SECTION_SPACING = 20
TRAIN_WIDTH = 40
TRAIN_HEIGHT = 20

class TrainDashboard:
    """A Pygame dashboard to visualize and manage train traffic."""
    
    def __init__(self, scenario_path="scenarios/scenario2.json", optimizer_path="optimizer_schedule.json", simulator_state_path="Optimizer/simulator_state.json"):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Train Traffic Management Dashboard")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 30)
        self.small_font = pygame.font.Font(None, 22)
        
        self.scenario_path = scenario_path
        self.optimizer_path = optimizer_path
        self.simulator_state_path = simulator_state_path
        
        self.sections: Dict[str, Any] = {}
        self.trains: Dict[str, Any] = {}
        self.schedule: List[Dict[str, Any]] = []
        self.optimizer_metadata: Dict[str, Any] = {}
        
        self.buttons: Dict[str, Tuple[pygame.Rect, Dict]] = {}
        self.section_rects: Dict[str, pygame.Rect] = {}
        self.running = True

    def load_and_parse_data(self):
        """Loads all data sources and updates the dashboard state."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        try:
            state_file = os.path.join(script_dir, self.simulator_state_path)
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            
            self.sections = {b_id: {**b_info, 'id': b_id} for b_id, b_info in state_data.get('blocks', {}).items()}
            
            new_trains_data = {t_id: {**t_info, 'id': t_id} for t_id, t_info in state_data.get('trains', {}).items()}
            for train_id, train_data in new_trains_data.items():
                if train_id in self.trains and 'animation_progress' in self.trains[train_id]:
                    if self.trains[train_id].get('current_block') != train_data.get('current_block'):
                        train_data['animation_progress'] = 0.0
                    else:
                        train_data['animation_progress'] = self.trains[train_id]['animation_progress']
                else:
                    train_data['animation_progress'] = 0.0
            self.trains = new_trains_data
            
            print(f"Successfully loaded simulator state from {self.simulator_state_path}")
        except Exception as e:
            print(f"Warning: Could not load simulator state: {e}")

        try:
            opt_file = os.path.join(script_dir, self.optimizer_path)
            with open(opt_file, 'r') as f:
                opt_data = json.load(f)
            self.schedule = opt_data.get('train_actions', [])
            self.optimizer_metadata = opt_data.get('optimization_metadata', {})
            print(f"Successfully loaded optimizer schedule from {self.optimizer_path}")
        except Exception as e:
            print(f"Warning: Optimizer schedule not found: {e}")
            self.schedule = []
            self.optimizer_metadata = {}
            
        self._update_ui_layout()

    def _update_ui_layout(self):
        """Recreates UI element positions after data load."""
        self.buttons.clear()
        self.section_rects.clear()

        start_x = 50
        start_y = 80
        sorted_section_ids = sorted(self.sections.keys())
        for i, section_id in enumerate(sorted_section_ids):
            x = start_x + i * (SECTION_WIDTH + SECTION_SPACING)
            self.section_rects[section_id] = pygame.Rect(x, start_y, SECTION_WIDTH, SECTION_HEIGHT)
            
        panel_rect = pygame.Rect(50, 300, WINDOW_WIDTH - 100, WINDOW_HEIGHT - 350)
        start_y = panel_rect.y + 90
        for i, item in enumerate(self.schedule):
            y = start_y + i * 50
            accept_rect = pygame.Rect(panel_rect.right - 2 * (BUTTON_WIDTH + 10), y - 5, BUTTON_WIDTH, BUTTON_HEIGHT)
            override_rect = pygame.Rect(panel_rect.right - (BUTTON_WIDTH + 10), y - 5, BUTTON_WIDTH, BUTTON_HEIGHT)
            self.buttons[f"accept_{i}"] = (accept_rect, item)
            self.buttons[f"override_{i}"] = (override_rect, item)

    def update(self, dt):
        """Update game state, including animations."""
        animation_speed = 0.2
        for train in self.trains.values():
            if train.get('state') == 'moving':
                train['animation_progress'] = min(1.0, train['animation_progress'] + animation_speed * dt)
            else:
                train['animation_progress'] = 0.0

    def draw_track_sections(self):
        """Draws the main track visualization."""
        for section_id, rect in self.section_rects.items():
            data = self.sections.get(section_id, {'is_free': True})
            color = GREEN if data['is_free'] else RED
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
            pygame.draw.rect(self.screen, WHITE, rect, 2, border_radius=5)
            
            id_text = self.font.render(section_id, True, WHITE)
            self.screen.blit(id_text, id_text.get_rect(center=rect.center))

    def draw_trains(self):
        """Draws the trains on the tracks, animated."""
        for train_id, train_data in self.trains.items():
            current_section_id = train_data.get('current_block')
            if not current_section_id or current_section_id not in self.section_rects:
                continue

            section_rect = self.section_rects[current_section_id]
            progress = train_data.get('animation_progress', 0.0)
            
            train_x = section_rect.left + (progress * (section_rect.width - TRAIN_WIDTH))
            train_y = section_rect.centery - (TRAIN_HEIGHT / 2)
            
            train_rect = pygame.Rect(train_x, train_y, TRAIN_WIDTH, TRAIN_HEIGHT)
            pygame.draw.rect(self.screen, TRAIN_COLOR, train_rect, border_radius=3)
            
            train_id_text = self.small_font.render(train_id, True, BLACK)
            self.screen.blit(train_id_text, train_id_text.get_rect(center=train_rect.center))

    def draw_train_states(self):
        """Draws the train information panel."""
        panel_rect = pygame.Rect(50, 160, WINDOW_WIDTH - 100, 120)
        pygame.draw.rect(self.screen, DARK_GRAY, panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, panel_rect, 2, border_radius=10)

        title_text = self.font.render("Train States", True, WHITE)
        self.screen.blit(title_text, (panel_rect.x + 20, panel_rect.y + 15))
        
        start_x = panel_rect.x + 20
        start_y = panel_rect.y + 50
        col_width = (panel_rect.width - 40) // 2
        
        for i, (train_id, train_data) in enumerate(self.trains.items()):
            col = i % 2
            row = i // 2
            x = start_x + col * col_width
            y = start_y + row * 25
            
            state = train_data.get('state', 'unknown')
            current_section = train_data.get('current_block', 'N/A')
            train_info = f"🚂 {train_id} - {state} @ {current_section}"
            
            train_text = self.small_font.render(train_info, True, WHITE)
            self.screen.blit(train_text, (x, y))

    def draw_schedule_panel(self):
        """Draws the interactive schedule management panel."""
        panel_rect = pygame.Rect(50, 300, WINDOW_WIDTH - 100, WINDOW_HEIGHT - 350)
        pygame.draw.rect(self.screen, DARK_GRAY, panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, panel_rect, 2, border_radius=10)

        title_text = self.font.render("Optimizer Schedule", True, WHITE)
        self.screen.blit(title_text, (panel_rect.x + 20, panel_rect.y + 20))
        
        if self.optimizer_metadata:
            meta_text = (f"Congestion: {self.optimizer_metadata.get('congestion_level', 0):.1%} | "
                         f"Waiting Trains: {self.optimizer_metadata.get('waiting_trains', 0)}")
            meta_surface = self.small_font.render(meta_text, True, LIGHT_GRAY)
            self.screen.blit(meta_surface, (panel_rect.x + 20, panel_rect.y + 50))
        
        if not self.schedule:
            no_data_text = self.small_font.render("Waiting for optimizer_schedule.json...", True, YELLOW)
            self.screen.blit(no_data_text, (panel_rect.x + 20, panel_rect.y + 90))
            return

        for i, item in enumerate(self.schedule):
            y = panel_rect.y + 90 + i * 50
            info = (f"Train: {item['train_id']} | Action: {item.get('action_type', 'N/A')} | "
                    f"Target: {item.get('target_block', 'N/A')}")
            info_text = self.small_font.render(info, True, WHITE)
            self.screen.blit(info_text, (panel_rect.x + 20, y))
            
            accept_rect, _ = self.buttons[f"accept_{i}"]
            override_rect, _ = self.buttons[f"override_{i}"]
            
            pygame.draw.rect(self.screen, GREEN, accept_rect, border_radius=5)
            accept_text = self.small_font.render("Accept", True, BLACK)
            self.screen.blit(accept_text, accept_text.get_rect(center=accept_rect.center))
            
            pygame.draw.rect(self.screen, YELLOW, override_rect, border_radius=5)
            override_text = self.small_font.render("Override", True, BLACK)
            self.screen.blit(override_text, override_text.get_rect(center=override_rect.center))

    def handle_mouse_click(self, pos: Tuple[int, int]):
        """Handles clicks on interactive UI elements."""
        for name, (rect, schedule_item) in self.buttons.items():
            if rect.collidepoint(pos):
                action_type, _ = name.split('_')
                train_id = schedule_item['train_id']
                
                if action_type == 'accept':
                    print(f"Action ACCEPTED for train {train_id}")
                elif action_type == 'override':
                    current_action = schedule_item.get('action_type', 'move_to_block')
                    overridden_action = 'hold' if current_action == 'move_to_block' else 'move_to_block'
                    feedback = {"user_feedback": {"train_id": train_id, "overridden_action": overridden_action}}
                    print("\n--- OVERRIDE SENT ---")
                    print(json.dumps(feedback, indent=2))
                    print("---------------------\n")
                return

    def run(self):
        """Main application loop."""
        print("Starting Dashboard... Press 'R' to reload data, 'ESC' to quit.")
        self.load_and_parse_data()
        
        while self.running:
            dt = self.clock.tick(60) / 1000.0

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
                    if event.button == 1:
                        self.handle_mouse_click(event.pos)
            
            self.update(dt)
            
            self.screen.fill(BLACK)
            self.draw_track_sections()
            self.draw_train_states()
            self.draw_schedule_panel()
            self.draw_trains()
            
            pygame.display.flip()

        pygame.quit()
        print("Dashboard closed.")

def main():
    dashboard = TrainDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()