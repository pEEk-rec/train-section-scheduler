#!/usr/bin/env python3
"""
Enhanced Train Traffic Management Dashboard
- Improved UI/UX: legend, tooltips, schedule scrolling, status feedback,
  wrapping section layout, hover highlighting of target blocks, smoother animation.
- Single-file Pygame dashboard. Drop into your repo and run.
"""

import pygame
import json
import os
import math
from typing import Dict, List, Any, Tuple

# --- Constants ---
BLACK = (20, 20, 30)
WHITE = (230, 230, 230)
GREEN = (40, 167, 69)
RED = (220, 53, 69)
DARK_GRAY = (52, 58, 64)
LIGHT_GRAY = (150, 150, 150)
YELLOW = (255, 193, 7)
TRAIN_COLOR = (0, 191, 255)
BLUE = (59, 130, 246)
ORANGE = (255, 140, 0)

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
PANEL_PADDING = 16

# Animation settings
ANIMATION_SPEED_BASE = 0.9  # per second normalized progress
ANIMATION_EASING = True

# Schedule scroll
SCHEDULE_ROW_H = 50


def lerp(a, b, t):
    return a + (b - a) * t


def ease_out_quad(t):
    return 1 - (1 - t) * (1 - t)


class TrainDashboard:
    """A Pygame dashboard to visualize and manage train traffic."""

    def __init__(self, scenario_path="scenarios/scenario2.json", optimizer_path="optimizer_schedule.json", simulator_state_path="Optimizer/simulator_state.json"):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("Train Traffic Management Dashboard")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 30)
        self.small_font = pygame.font.Font(None, 20)
        self.tiny_font = pygame.font.Font(None, 16)

        self.scenario_path = scenario_path
        self.optimizer_path = optimizer_path
        self.simulator_state_path = simulator_state_path

        self.sections: Dict[str, Any] = {}
        self.trains: Dict[str, Any] = {}
        self.schedule: List[Dict[str, Any]] = []
        self.optimizer_metadata: Dict[str, Any] = {}

        # UI bookkeeping
        self.buttons: Dict[str, Tuple[pygame.Rect, Dict]] = {}
        self.section_rects: Dict[str, pygame.Rect] = {}
        self.running = True
        self.schedule_scroll = 0
        self.selected_schedule_index = None
        self.hovered_section = None
        self.hovered_train = None
        self.mouse_pos = (0, 0)

        # responsive layout values
        self.sections_area = pygame.Rect(50, 80, WINDOW_WIDTH - 100, 180)
        self.schedule_panel_rect = pygame.Rect(50, 300, WINDOW_WIDTH - 100, WINDOW_HEIGHT - 350)

        # For smoother updates: track animation targets
        # trains[*]['animation_progress'] is 0..1 for current block

    # ---------------- Data Loading ----------------
    def load_and_parse_data(self):
        """Loads all data sources and updates the dashboard state."""
        script_dir = os.path.dirname(os.path.abspath(__file__))

        try:
            state_file = os.path.join(script_dir, self.simulator_state_path)
            with open(state_file, 'r') as f:
                state_data = json.load(f)

            # blocks/sections
            self.sections = {b_id: {**b_info, 'id': b_id} for b_id, b_info in state_data.get('blocks', {}).items()}

            # trains
            new_trains_data = {t_id: {**t_info, 'id': t_id} for t_id, t_info in state_data.get('trains', {}).items()}
            for train_id, train_data in new_trains_data.items():
                # preserve animation progress if same block else reset
                if train_id in self.trains and 'animation_progress' in self.trains[train_id]:
                    if self.trains[train_id].get('current_block') != train_data.get('current_block'):
                        train_data['animation_progress'] = 0.0
                    else:
                        train_data['animation_progress'] = self.trains[train_id].get('animation_progress', 0.0)
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
            # extend schedule items with UI status if missing
            raw_schedule = opt_data.get('train_actions', [])
            self.schedule = []
            for item in raw_schedule:
                item_copy = dict(item)
                item_copy.setdefault('status', 'pending')  # pending, accepted, overridden
                item_copy.setdefault('ui_id', f"sched_{len(self.schedule)}")
                self.schedule.append(item_copy)

            self.optimizer_metadata = opt_data.get('optimization_metadata', {})
            print(f"Successfully loaded optimizer schedule from {self.optimizer_path}")
        except Exception as e:
            print(f"Warning: Optimizer schedule not found: {e}")
            self.schedule = []
            self.optimizer_metadata = {}

        self._update_ui_layout()

    # ---------------- Layout ----------------
    def _update_ui_layout(self):
        """Recreates UI element positions after data load or window resize."""
        self.buttons.clear()
        self.section_rects.clear()

        # sections area - wrap rows to fit width
        margin = 50
        spacing_x = SECTION_SPACING
        spacing_y = 20
        area_x = margin
        area_y = 80
        max_w = self.screen.get_width() - 2 * margin
        x = area_x
        y = area_y
        row_height = SECTION_HEIGHT

        sorted_section_ids = sorted(self.sections.keys())
        for i, section_id in enumerate(sorted_section_ids):
            if x + SECTION_WIDTH > area_x + max_w:
                # wrap to next row
                x = area_x
                y += row_height + spacing_y
            rect = pygame.Rect(x, y, SECTION_WIDTH, SECTION_HEIGHT)
            self.section_rects[section_id] = rect
            x += SECTION_WIDTH + spacing_x

        # schedule panel
        panel_h = max(200, self.screen.get_height() - 350)
        self.schedule_panel_rect = pygame.Rect(50, 300, self.screen.get_width() - 100, panel_h)

        # compute buttons for each schedule row (positions relative to panel + scroll)
        for i, item in enumerate(self.schedule):
            y_row = self.schedule_panel_rect.y + 90 + i * SCHEDULE_ROW_H
            accept_rect = pygame.Rect(self.schedule_panel_rect.right - 2 * (BUTTON_WIDTH + 10), y_row - 5, BUTTON_WIDTH, BUTTON_HEIGHT)
            override_rect = pygame.Rect(self.schedule_panel_rect.right - (BUTTON_WIDTH + 10), y_row - 5, BUTTON_WIDTH, BUTTON_HEIGHT)
            self.buttons[f"accept_{i}"] = (accept_rect, item)
            self.buttons[f"override_{i}"] = (override_rect, item)

    # ---------------- Update / Animation ----------------
    def update(self, dt):
        """Update game state, including animations."""
        # dynamic speed: maybe depends on train property in future
        for train in self.trains.values():
            state = train.get('state')
            speed = train.get('speed', 1.0)
            animation_speed = ANIMATION_SPEED_BASE * (speed if isinstance(speed, (int, float)) else 1.0)

            if state == 'moving':
                progress = train.get('animation_progress', 0.0)
                progress = min(1.0, progress + animation_speed * dt)
                train['animation_progress'] = progress
            else:
                # keep last frame but don't progress
                train.setdefault('animation_progress', 0.0)

    # ---------------- Drawing helpers ----------------
    def draw_legend(self):
        """Draws a small legend explaining colors and states."""
        rect = pygame.Rect(self.screen.get_width() - 260, 20, 210, 110)
        pygame.draw.rect(self.screen, DARK_GRAY, rect, border_radius=8)
        pygame.draw.rect(self.screen, WHITE, rect, 2, border_radius=8)

        lines = [
            (GREEN, "Free Section"),
            (RED, "Occupied Section"),
            (TRAIN_COLOR, "Train"),
            (YELLOW, "Override Button"),
            (GREEN, "Accept Button"),
        ]
        for i, (col, text) in enumerate(lines):
            dot = pygame.Rect(rect.x + 12, rect.y + 12 + i * 20, 12, 12)
            pygame.draw.rect(self.screen, col, dot)
            t = self.tiny_font.render(text, True, WHITE)
            self.screen.blit(t, (dot.right + 8, dot.y - 2))

    def draw_track_sections(self):
        """Draws the main track visualization."""
        for section_id, rect in self.section_rects.items():
            data = self.sections.get(section_id, {'is_free': True})
            is_free = data.get('is_free', True)
            color = GREEN if is_free else RED

            # highlight if this is target of selected schedule or hovered schedule
            highlight = False
            if self.selected_schedule_index is not None and 0 <= self.selected_schedule_index < len(self.schedule):
                sel_item = self.schedule[self.selected_schedule_index]
                if sel_item.get('target_block') == section_id:
                    highlight = True
            # if mouse hovering schedule row, also highlight
            # if current hovered schedule matches target
            if self.hovered_section == section_id:
                highlight = True

            # draw
            pygame.draw.rect(self.screen, color, rect, border_radius=6)
            border_col = BLUE if highlight else WHITE
            pygame.draw.rect(self.screen, border_col, rect, 3 if highlight else 2, border_radius=6)

            id_text = self.small_font.render(section_id, True, WHITE)
            self.screen.blit(id_text, id_text.get_rect(center=rect.center))

    def draw_trains(self):
        """Draws the trains on the tracks, animated."""
        for train_id, train_data in self.trains.items():
            current_section_id = train_data.get('current_block')
            if not current_section_id or current_section_id not in self.section_rects:
                continue

            section_rect = self.section_rects[current_section_id]
            progress = train_data.get('animation_progress', 0.0)
            if ANIMATION_EASING:
                progress = ease_out_quad(progress)

            # position the train along the section width
            train_x = lerp(section_rect.left + 6, section_rect.right - TRAIN_WIDTH - 6, progress)
            train_y = section_rect.centery - (TRAIN_HEIGHT / 2)

            train_rect = pygame.Rect(train_x, train_y, TRAIN_WIDTH, TRAIN_HEIGHT)
            pygame.draw.rect(self.screen, TRAIN_COLOR, train_rect, border_radius=4)

            train_id_text = self.tiny_font.render(train_id, True, BLACK)
            self.screen.blit(train_id_text, train_id_text.get_rect(center=train_rect.center))

            # if hovering train show small halo
            if self.hovered_train == train_id:
                pygame.draw.rect(self.screen, ORANGE, train_rect.inflate(8, 8), 2, border_radius=6)

    def draw_train_states(self):
        """Draws the train information panel."""
        panel_rect = pygame.Rect(50, 160, self.screen.get_width() - 100, 120)
        pygame.draw.rect(self.screen, DARK_GRAY, panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, panel_rect, 2, border_radius=10)

        title_text = self.font.render("Train States", True, WHITE)
        self.screen.blit(title_text, (panel_rect.x + 20, panel_rect.y + 12))

        start_x = panel_rect.x + 20
        start_y = panel_rect.y + 48
        col_width = (panel_rect.width - 40) // 2

        for i, (train_id, train_data) in enumerate(self.trains.items()):
            col = i % 2
            row = i // 2
            x = start_x + col * col_width
            y = start_y + row * 25

            state = train_data.get('state', 'unknown')
            current_section = train_data.get('current_block', 'N/A')
            info = f"{train_id}: {state} @ {current_section}"
            train_text = self.small_font.render(info, True, WHITE)
            self.screen.blit(train_text, (x, y))

    def draw_schedule_panel(self):
        """Draws the interactive schedule management panel with scrolling."""
        panel_rect = self.schedule_panel_rect
        pygame.draw.rect(self.screen, DARK_GRAY, panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, WHITE, panel_rect, 2, border_radius=10)

        title_text = self.font.render("Optimizer Schedule", True, WHITE)
        self.screen.blit(title_text, (panel_rect.x + 20, panel_rect.y + 12))

        if self.optimizer_metadata:
            meta_text = (f"Congestion: {self.optimizer_metadata.get('congestion_level', 0):.1%} | "
                         f"Waiting Trains: {self.optimizer_metadata.get('waiting_trains', 0)}")
            meta_surface = self.small_font.render(meta_text, True, LIGHT_GRAY)
            self.screen.blit(meta_surface, (panel_rect.x + 20, panel_rect.y + 48))

        if not self.schedule:
            no_data_text = self.small_font.render("Waiting for optimizer_schedule.json...", True, YELLOW)
            self.screen.blit(no_data_text, (panel_rect.x + 20, panel_rect.y + 90))
            return

        # visible region clipping
        clip = self.screen.get_clip()
        self.screen.set_clip(panel_rect.inflate(-10, -10))

        # draw rows with scroll offset
        y0 = panel_rect.y + 90 - self.schedule_scroll
        for i, item in enumerate(self.schedule):
            row_rect = pygame.Rect(panel_rect.x + 10, y0 + i * SCHEDULE_ROW_H, panel_rect.width - 40, SCHEDULE_ROW_H - 10)

            # detect if row is visible before drawing heavy stuff
            if row_rect.bottom < panel_rect.y + 80 or row_rect.top > panel_rect.bottom - 10:
                continue

            # background based on status
            status = item.get('status', 'pending')
            if status == 'accepted':
                bg = (30, 70, 30)
            elif status == 'overridden':
                bg = (70, 50, 10)
            else:
                bg = (40, 40, 40)

            pygame.draw.rect(self.screen, bg, row_rect, border_radius=6)
            pygame.draw.rect(self.screen, WHITE, row_rect, 1, border_radius=6)

            info = (f"Train: {item.get('train_id')} | Action: {item.get('action_type', 'N/A')} | "
                    f"Target: {item.get('target_block', 'N/A')}")
            info_text = self.small_font.render(info, True, WHITE)
            self.screen.blit(info_text, (row_rect.x + 12, row_rect.y + 8))

            # buttons (compute dynamic positions since panel can be resized)
            accept_rect = pygame.Rect(row_rect.right - 2 * (BUTTON_WIDTH + 10), row_rect.y + 8, BUTTON_WIDTH, BUTTON_HEIGHT)
            override_rect = pygame.Rect(row_rect.right - (BUTTON_WIDTH + 10), row_rect.y + 8, BUTTON_WIDTH, BUTTON_HEIGHT)
            # store updated button rects for click testing
            self.buttons[f"accept_{i}"] = (accept_rect, item)
            self.buttons[f"override_{i}"] = (override_rect, item)

            # draw accept/override but disabled if already acted upon
            if status == 'pending':
                pygame.draw.rect(self.screen, GREEN, accept_rect, border_radius=6)
                pygame.draw.rect(self.screen, YELLOW, override_rect, border_radius=6)
            else:
                # dimmed
                pygame.draw.rect(self.screen, (80, 120, 80), accept_rect, border_radius=6)
                pygame.draw.rect(self.screen, (120, 100, 60), override_rect, border_radius=6)

            accept_text = self.small_font.render("Accept", True, BLACK)
            self.screen.blit(accept_text, accept_text.get_rect(center=accept_rect.center))
            override_text = self.small_font.render("Override", True, BLACK)
            self.screen.blit(override_text, override_text.get_rect(center=override_rect.center))

            # subtle indicator for selected/hovered row
            if i == self.selected_schedule_index:
                pygame.draw.rect(self.screen, BLUE, row_rect, 2, border_radius=6)

        # restore clip
        self.screen.set_clip(clip)

        # scrollbar visual
        total_h = len(self.schedule) * SCHEDULE_ROW_H
        view_h = panel_rect.height - 110
        if total_h > 0 and total_h > view_h:
            scrollbar_h = max(24, int(view_h * (view_h / total_h)))
            scroll_max = max(1, total_h - view_h)
            scroll_ratio = self.schedule_scroll / scroll_max
            sb_x = panel_rect.right - 14
            sb_y = panel_rect.y + 90 + int((view_h - scrollbar_h) * scroll_ratio)
            pygame.draw.rect(self.screen, LIGHT_GRAY, pygame.Rect(sb_x, sb_y, 8, scrollbar_h), border_radius=4)

    def draw_tooltip(self):
        # show tooltip near mouse for hovered section or train
        mx, my = self.mouse_pos
        tooltip_lines = []
        if self.hovered_section and self.hovered_section in self.sections:
            s = self.sections[self.hovered_section]
            tooltip_lines.append(f"Section: {self.hovered_section}")
            tooltip_lines.append(f"Free: {s.get('is_free', True)}")
            next_train = s.get('next_train', 'N/A')
            tooltip_lines.append(f"Next Train: {next_train}")
        elif self.hovered_train and self.hovered_train in self.trains:
            t = self.trains[self.hovered_train]
            tooltip_lines.append(f"Train: {self.hovered_train}")
            tooltip_lines.append(f"State: {t.get('state', 'N/A')}")
            tooltip_lines.append(f"Block: {t.get('current_block', 'N/A')}")
            speed = t.get('speed', 'N/A')
            tooltip_lines.append(f"Speed: {speed}")

        if not tooltip_lines:
            return

        # render tooltip
        padding = 8
        surfaces = [self.tiny_font.render(l, True, WHITE) for l in tooltip_lines]
        w = max(s.get_width() for s in surfaces) + padding * 2
        h = sum(s.get_height() for s in surfaces) + padding * 2
        rect = pygame.Rect(mx + 16, my + 16, w, h)
        pygame.draw.rect(self.screen, DARK_GRAY, rect, border_radius=6)
        pygame.draw.rect(self.screen, WHITE, rect, 1, border_radius=6)
        y = rect.y + padding
        for s in surfaces:
            self.screen.blit(s, (rect.x + padding, y))
            y += s.get_height()

    # ---------------- Input / Interaction ----------------
    def handle_mouse_click(self, pos: Tuple[int, int]):
        """Handles clicks on interactive UI elements."""
        # first, check schedule buttons
        for name, (rect, schedule_item) in list(self.buttons.items()):
            # ensure button rect is valid and clickable (some button rects are for schedule rows)
            if rect and rect.collidepoint(pos):
                if name.startswith('accept_'):
                    idx = int(name.split('_')[1])
                    self._accept_schedule(idx)
                    return
                elif name.startswith('override_'):
                    idx = int(name.split('_')[1])
                    self._override_schedule(idx)
                    return

        # click on section to select it's id
        for section_id, rect in self.section_rects.items():
            if rect.collidepoint(pos):
                self.selected_schedule_index = None
                # attempt to find schedule items targeting this section and select the first
                for i, item in enumerate(self.schedule):
                    if item.get('target_block') == section_id:
                        self.selected_schedule_index = i
                        break
                return

        # click on train to select schedule entries for that train
        for train_id in self.trains.keys():
            # rough check: find train rect and collide
            train_data = self.trains[train_id]
            cb = train_data.get('current_block')
            if cb in self.section_rects:
                rect = self.section_rects[cb]
                progress = ease_out_quad(train_data.get('animation_progress', 0.0))
                tx = lerp(rect.left + 6, rect.right - TRAIN_WIDTH - 6, progress)
                train_rect = pygame.Rect(tx, rect.centery - TRAIN_HEIGHT / 2, TRAIN_WIDTH, TRAIN_HEIGHT)
                if train_rect.collidepoint(pos):
                    # select schedule items for this train
                    for i, item in enumerate(self.schedule):
                        if item.get('train_id') == train_id:
                            self.selected_schedule_index = i
                            return

    def _accept_schedule(self, idx: int):
        if idx < 0 or idx >= len(self.schedule):
            return
        item = self.schedule[idx]
        if item.get('status') != 'pending':
            print(f"Schedule item {idx} already {item.get('status')}")
            return
        item['status'] = 'accepted'
        print(f"Action ACCEPTED for train {item.get('train_id')}")

    def _override_schedule(self, idx: int):
        if idx < 0 or idx >= len(self.schedule):
            return
        item = self.schedule[idx]
        if item.get('status') != 'pending':
            print(f"Schedule item {idx} already {item.get('status')}")
            return
        current_action = item.get('action_type', 'move_to_block')
        overridden_action = 'hold' if current_action == 'move_to_block' else 'move_to_block'
        item['status'] = 'overridden'
        item['overridden_action'] = overridden_action
        feedback = {"user_feedback": {"train_id": item.get('train_id'), "overridden_action": overridden_action}}
        print("\n--- OVERRIDE SENT ---")
        print(json.dumps(feedback, indent=2))
        print("---------------------\n")

    # ---------------- Event Loop ----------------
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
                    elif event.button == 4:  # wheel up -> scroll up
                        self.schedule_scroll = max(0, self.schedule_scroll - 30)
                    elif event.button == 5:  # wheel down -> scroll down
                        # compute maximum scroll
                        total_h = len(self.schedule) * SCHEDULE_ROW_H
                        view_h = self.schedule_panel_rect.height - 110
                        max_scroll = max(0, total_h - view_h)
                        self.schedule_scroll = min(max_scroll, self.schedule_scroll + 30)
                elif event.type == pygame.MOUSEMOTION:
                    self.mouse_pos = event.pos
                    self._update_hover_states(event.pos)
                elif event.type == pygame.VIDEORESIZE:
                    # reposition elements and keep proportions
                    w, h = event.size
                    pygame.display.set_mode((w, h), pygame.RESIZABLE)
                    self._update_ui_layout()

            self.update(dt)

            # draw
            self.screen.fill(BLACK)
            self.draw_track_sections()
            self.draw_train_states()
            self.draw_schedule_panel()
            self.draw_trains()
            self.draw_legend()
            self.draw_tooltip()

            pygame.display.flip()

        pygame.quit()
        print("Dashboard closed.")

    # ---------------- Hover logic ----------------
    def _update_hover_states(self, pos: Tuple[int, int]):
        mx, my = pos
        self.hovered_section = None
        self.hovered_train = None
        # check schedule rows hover -> highlight target
        self.selected_schedule_index = None
        panel = self.schedule_panel_rect
        if panel.collidepoint(pos):
            # compute which row is under mouse considering scroll
            local_y = my - (panel.y + 90) + self.schedule_scroll
            if local_y >= 0:
                idx = int(local_y // SCHEDULE_ROW_H)
                if 0 <= idx < len(self.schedule):
                    self.selected_schedule_index = idx
                    # highlight the target block of that schedule
                    target = self.schedule[idx].get('target_block')
                    if target in self.section_rects:
                        self.hovered_section = target

        # check section hover
        for section_id, rect in self.section_rects.items():
            if rect.collidepoint(pos):
                self.hovered_section = section_id
                break

        # check train hover (overrides section hover for train)
        for train_id, train in self.trains.items():
            cb = train.get('current_block')
            if not cb or cb not in self.section_rects:
                continue
            rect = self.section_rects[cb]
            prog = ease_out_quad(train.get('animation_progress', 0.0))
            tx = lerp(rect.left + 6, rect.right - TRAIN_WIDTH - 6, prog)
            train_rect = pygame.Rect(tx, rect.centery - TRAIN_HEIGHT / 2, TRAIN_WIDTH, TRAIN_HEIGHT)
            if train_rect.collidepoint(pos):
                self.hovered_train = train_id
                self.hovered_section = None
                break


def main():
    dashboard = TrainDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
