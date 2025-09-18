# Train Traffic Management Dashboard

A Pygame-based visualization dashboard for monitoring railway network state and train management.

## Features

- **Real-time Visualization**: Displays track sections with color-coded status (green = free, red = occupied)
- **Train Monitoring**: Shows current train states and positions
- **Interactive Interface**: Click on track sections for detailed information
- **Data Integration**: Reads from simulator output JSON files
- **Future-Ready**: Prepared for optimizer integration

## Requirements

- Python 3.7+
- Pygame 2.0.0+

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have simulation data:
```bash
# Run the simulator to generate data
python Simulator/run_sim.py
```

## Usage

Run the dashboard:
```bash
python dashboard.py
```

### Controls

- **Mouse Click**: Click on track sections to see details
- **R Key**: Reload data from JSON files
- **ESC Key**: Exit the dashboard

## Data Sources

The dashboard reads from:
- `logs/run1.json` - Simulator output with event logs
- `scenarios/scenario2.json` - Track and train configuration
- `optimizer_output.json` - Future optimizer output (not yet implemented)

## Visual Layout

- **Top Bar**: Status information and data source indicators
- **Track Sections**: Horizontal layout showing section status and current trains
- **Train States Panel**: List of all trains with their current status
- **Schedule Panel**: Placeholder for future optimizer integration

## Future Enhancements

When the optimizer is implemented, the dashboard will support:
- Schedule display with Accept/Override buttons
- User feedback collection
- Real-time schedule updates
- Interactive train control

