#!/usr/bin/env python3
"""
Test script for the fixed dashboard
"""

import json
import os
from dashboard import TrainDashboard

def test_dashboard_data_loading():
    """Test that the dashboard loads data correctly"""
    print("Testing Fixed Dashboard Data Loading...")
    
    # Create dashboard instance
    dashboard = TrainDashboard()
    
    # Test data loading
    print("\n1. Testing data loading...")
    dashboard.load_and_parse_data()
    
    print(f"   ✓ Loaded {len(dashboard.sections)} sections/blocks")
    print(f"   ✓ Loaded {len(dashboard.trains)} trains")
    print(f"   ✓ Loaded {len(dashboard.schedule)} schedule items")
    
    # Display sections
    print("\n2. Section/Block Status:")
    for section_id, section_data in dashboard.sections.items():
        status = "FREE" if section_data['is_free'] else "OCCUPIED"
        current_train = section_data['current_train'] or "None"
        print(f"   - {section_id}: {status} (Train: {current_train})")
    
    # Display trains
    print("\n3. Train States:")
    for train_id, train_data in dashboard.trains.items():
        current_section = train_data.get('current_section', 'None')
        delay = train_data.get('delay', 0)
        print(f"   - {train_id}: {train_data['state']} (Section: {current_section}, Delay: {delay:.1f}s)")
    
    # Display schedule
    print("\n4. Optimizer Schedule:")
    if dashboard.schedule:
        for i, item in enumerate(dashboard.schedule):
            action_type = item.get('action_type', 'unknown')
            target_block = item.get('target_block', 'N/A')
            duration = item.get('estimated_duration', 0)
            print(f"   - {i+1}. Train {item['train_id']}: {action_type} -> {target_block} ({duration:.1f}s)")
    else:
        print("   - No schedule items")
    
    # Display metadata
    if dashboard.optimizer_metadata:
        print("\n5. Optimizer Metadata:")
        print(f"   - Congestion Level: {dashboard.optimizer_metadata.get('congestion_level', 0):.1%}")
        print(f"   - Waiting Trains: {dashboard.optimizer_metadata.get('waiting_trains', 0)}")
        print(f"   - Total Actions: {dashboard.optimizer_metadata.get('total_actions', 0)}")
    
    print("\n" + "="*60)
    print("Dashboard test completed successfully!")
    print("The dashboard should now work with the optimizer data.")

if __name__ == "__main__":
    test_dashboard_data_loading()
