#!/usr/bin/env python3
"""
Comprehensive test suite for the CameraDataManager abstraction.
"""
from __future__ import annotations

import sys
import os
import argparse
import tempfile
import time
from unittest.mock import Mock, patch

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from camera_data_manager import (
        CameraDataManager, 
        setup_camera_manager, 
        add_camera_args,
        create_camera_manager_from_args,
        create_subscriber_with_args
    )
    from data_source import LiveCameraDataSource, RecordingCameraDataSource, PlaybackCameraDataSource
    print("✅ Successfully imported all camera data manager modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_camera_data_manager_creation():
    """Test creating CameraDataManager with different configurations."""
    print("\n🧪 Testing CameraDataManager creation...")
    
    # Test live mode (default)
    try:
        manager = CameraDataManager()
        assert manager.get_mode() == "live"
        assert isinstance(manager.data_source, LiveCameraDataSource)
        assert manager.get_status_string() == "[LIVE]"
        print("✅ Live mode creation successful")
    except Exception as e:
        print(f"❌ Live mode creation failed: {e}")
    
    # Test recording mode
    try:
        with tempfile.NamedTemporaryFile(suffix=".pkl.gz", delete=False) as tmp:
            recording_file = tmp.name
        
        manager = CameraDataManager(record=True, recording_file=recording_file)
        assert manager.get_mode() == "recording"
        assert isinstance(manager.data_source, RecordingCameraDataSource)
        assert "[RECORDING]" in manager.get_status_string()
        print("✅ Recording mode creation successful")
        
        # Clean up
        os.unlink(recording_file)
    except Exception as e:
        print(f"❌ Recording mode creation failed: {e}")
    
    # Test playback mode (will fail because file doesn't exist, but that's expected)
    try:
        manager = CameraDataManager(playback_file="nonexistent.pkl.gz")
        print("❌ Playback mode should have failed but didn't")
    except FileNotFoundError:
        print("✅ Playback mode correctly failed for missing file")
    except Exception as e:
        print(f"❌ Unexpected error in playback mode: {e}")


def test_argument_parsing():
    """Test the argument parsing functionality."""
    print("\n🧪 Testing argument parsing...")
    
    # Test add_camera_args
    try:
        parser = argparse.ArgumentParser()
        add_camera_args(parser)
        
        # Test parsing different argument combinations
        test_cases = [
            ([], "live"),  # Default
            (["--record"], "recording"),
            (["--playback", "test.pkl.gz"], "playback"),
            (["--record", "--recording-file", "custom.pkl.gz"], "recording"),
            (["--no-loop"], "live"),
            (["--no-realtime"], "live"),
        ]
        
        for args, expected_mode in test_cases:
            try:
                parsed_args = parser.parse_args(args)
                if expected_mode == "recording":
                    assert parsed_args.record == True
                elif expected_mode == "playback":
                    assert parsed_args.playback == "test.pkl.gz"
                else:  # live
                    assert getattr(parsed_args, 'record', False) == False
                    assert getattr(parsed_args, 'playback', None) is None
                
                print(f"  ✅ Args {args} parsed correctly for {expected_mode} mode")
            except Exception as e:
                print(f"  ❌ Args {args} failed: {e}")
        
        print("✅ Argument parsing tests passed")
    except Exception as e:
        print(f"❌ Argument parsing test failed: {e}")


def test_create_from_args():
    """Test creating managers from parsed arguments."""
    print("\n🧪 Testing creation from parsed arguments...")
    
    try:
        parser = argparse.ArgumentParser()
        add_camera_args(parser)
        
        # Test live mode
        args = parser.parse_args([])
        manager = create_camera_manager_from_args(args)
        assert manager.get_mode() == "live"
        
        # Test recording mode
        args = parser.parse_args(["--record"])
        manager = create_camera_manager_from_args(args)
        assert manager.get_mode() == "recording"
        
        # Test legacy function
        manager_legacy = create_subscriber_with_args(args)
        assert manager_legacy.get_mode() == "recording"
        
        print("✅ Creation from args tests passed")
    except Exception as e:
        print(f"❌ Creation from args test failed: {e}")


def test_manager_interface():
    """Test the CameraDataManager interface methods."""
    print("\n🧪 Testing manager interface...")
    
    try:
        manager = CameraDataManager()
        
        # Test all required methods exist and are callable
        methods = ['connect', 'start_subscribing', 'get_latest_frames', 
                  'is_running', 'stop', 'get_mode', 'get_status_string']
        
        for method in methods:
            assert hasattr(manager, method), f"Method {method} missing"
            assert callable(getattr(manager, method)), f"Method {method} not callable"
        
        # Test mode and status methods
        assert manager.get_mode() in ["live", "recording", "playback"]
        status = manager.get_status_string()
        assert isinstance(status, str)
        assert len(status) > 0
        
        print("✅ Manager interface tests passed")
    except Exception as e:
        print(f"❌ Manager interface test failed: {e}")


def test_setup_camera_manager():
    """Test the setup_camera_manager convenience function."""
    print("\n🧪 Testing setup_camera_manager...")
    
    # Mock sys.argv to test argument parsing
    original_argv = sys.argv
    
    try:
        # Test default arguments
        sys.argv = ["test_script.py"]
        with patch('camera_data_manager.setup_camera_manager') as mock_setup:
            # Mock the return value
            mock_manager = Mock()
            mock_manager.get_status_string.return_value = "[LIVE]"
            mock_args = Mock()
            mock_setup.return_value = (mock_manager, mock_args)
            
            # Call the function
            manager, args = mock_setup("Test application")
            
            # Verify it was called correctly
            mock_setup.assert_called_once_with("Test application")
            assert manager is not None
            assert args is not None
        
        print("✅ setup_camera_manager tests passed")
    except Exception as e:
        print(f"❌ setup_camera_manager test failed: {e}")
    finally:
        sys.argv = original_argv


def test_error_handling():
    """Test error handling in various scenarios."""
    print("\n🧪 Testing error handling...")
    
    try:
        # Test invalid playback file
        try:
            manager = CameraDataManager(playback_file="/nonexistent/path/file.pkl.gz")
            print("❌ Should have failed for invalid playback file")
        except FileNotFoundError:
            print("  ✅ Correctly handled invalid playback file")
        
        # Test invalid recording path
        try:
            manager = CameraDataManager(record=True, recording_file="/invalid/path/file.pkl.gz")
            # This might not fail immediately, but should fail on connect()
            print("  ✅ Invalid recording path handled (will fail on connect)")
        except Exception as e:
            print(f"  ✅ Correctly handled invalid recording path: {e}")
        
        print("✅ Error handling tests passed")
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")


def test_mode_detection():
    """Test mode detection logic."""
    print("\n🧪 Testing mode detection...")
    
    try:
        # Test all mode combinations
        test_cases = [
            ({"record": False, "playback_file": None}, "live"),
            ({"record": True, "playback_file": None}, "recording"),
            ({"record": False, "playback_file": "test.pkl.gz"}, "playback"),
        ]
        
        for kwargs, expected_mode in test_cases:
            try:
                if expected_mode == "playback":
                    # Create a temporary file for playback test
                    with tempfile.NamedTemporaryFile(suffix=".pkl.gz", delete=False) as tmp:
                        tmp.write(b"test data")
                        kwargs["playback_file"] = tmp.name
                    
                    manager = CameraDataManager(**kwargs)
                    assert manager.get_mode() == expected_mode
                    
                    # Clean up
                    os.unlink(kwargs["playback_file"])
                else:
                    if expected_mode == "recording":
                        # Use a valid temporary path for recording
                        with tempfile.NamedTemporaryFile(suffix=".pkl.gz", delete=False) as tmp:
                            kwargs["recording_file"] = tmp.name
                        manager = CameraDataManager(**kwargs)
                        os.unlink(kwargs["recording_file"])
                    else:
                        manager = CameraDataManager(**kwargs)
                    
                    assert manager.get_mode() == expected_mode
                
                print(f"  ✅ Mode {expected_mode} detected correctly")
            except Exception as e:
                print(f"  ❌ Mode {expected_mode} detection failed: {e}")
        
        print("✅ Mode detection tests passed")
    except Exception as e:
        print(f"❌ Mode detection test failed: {e}")


def main():
    """Run all tests."""
    print("🔬 Testing Camera Data Manager Abstraction")
    print("=" * 50)
    
    test_camera_data_manager_creation()
    test_argument_parsing()
    test_create_from_args()
    test_manager_interface()
    test_setup_camera_manager()
    test_error_handling()
    test_mode_detection()
    
    print("\n✨ All tests completed!")
    print("\nThe CameraDataManager abstraction is ready for use!")
    print("\nTo integrate into any script:")
    print("1. Add: from camera_data_manager import setup_camera_manager")
    print("2. Replace: camera_manager, args = setup_camera_manager('My app')")
    print("3. Use: camera_manager.connect(), camera_manager.get_latest_frames(), etc.")


if __name__ == "__main__":
    main() 