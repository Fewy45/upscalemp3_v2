"""
Test script to verify the preprocessing pipeline worked correctly.
Run this after creating TFRecords to ensure everything is working.
"""

import os
import numpy as np
import tensorflow as tf
import soundfile as sf
import json
from pathlib import Path
import matplotlib.pyplot as plt
import wave
import io

def test_tfrecords(tfrecord_dir, num_files_to_test=3, num_clips_to_test=5):
    """
    Comprehensive test suite for verifying TFRecord creation.
    
    Args:
        tfrecord_dir: Directory containing TFRecord files
        num_files_to_test: Number of TFRecord files to test
        num_clips_to_test: Number of clips to test per file
    """
    print("="*60)
    print("TFRECORD PREPROCESSING VERIFICATION")
    print("="*60)
    
    # 1. Check if TFRecord files exist
    print("\n1. Checking TFRecord files...")
    tfrecord_files = list(Path(tfrecord_dir).glob("*.tfrecord"))
    
    if not tfrecord_files:
        print("❌ ERROR: No TFRecord files found in", tfrecord_dir)
        return False
    
    print(f"✅ Found {len(tfrecord_files)} TFRecord files")
    
    # 2. Check metadata file
    print("\n2. Checking metadata...")
    metadata_path = Path(tfrecord_dir) / "metadata.json"
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"✅ Metadata found:")
        print(f"   - Total clips: {metadata['total_clips']}")
        print(f"   - Sample rate: {metadata['sample_rate']}Hz")
        print(f"   - Clip duration: {metadata['clip_duration']}s")
        print(f"   - Datasets: {metadata['datasets']}")
    else:
        print("⚠️  No metadata.json found (optional but recommended)")
        metadata = None
    
    # 3. Test reading TFRecords
    print(f"\n3. Testing TFRecord reading (first {num_files_to_test} files)...")
    
    test_results = {
        'files_tested': 0,
        'clips_tested': 0,
        'sample_rates': [],
        'clip_lengths': [],
        'errors': []
    }
    
    # Feature description for parsing
    feature_description = {
        'audio_binary': tf.io.FixedLenFeature([], tf.string),
        'sample_rate': tf.io.FixedLenFeature([], tf.int64),
        'samples': tf.io.FixedLenFeature([], tf.int64),
        'source_file': tf.io.FixedLenFeature([], tf.string),
        'clip_index': tf.io.FixedLenFeature([], tf.int64),
        'dataset': tf.io.FixedLenFeature([], tf.string),
    }
    
    for tfrecord_file in tfrecord_files[:num_files_to_test]:
        print(f"\n   Testing {tfrecord_file.name}...")
        
        try:
            # Create dataset
            dataset = tf.data.TFRecordDataset(
                str(tfrecord_file),
                compression_type='GZIP'
            )
            
            clips_in_file = 0
            for raw_record in dataset.take(num_clips_to_test):
                try:
                    # Parse the record
                    example = tf.train.Example()
                    example.ParseFromString(raw_record.numpy())
                    
                    # Parse features
                    parsed = tf.io.parse_single_example(raw_record, feature_description)
                    
                    # Get metadata
                    sample_rate = parsed['sample_rate'].numpy()
                    num_samples = parsed['samples'].numpy()
                    source_file = parsed['source_file'].numpy().decode('utf-8')
                    clip_index = parsed['clip_index'].numpy()
                    dataset_name = parsed['dataset'].numpy().decode('utf-8')
                    
                    # Decode audio
                    audio_tensor = tf.audio.decode_wav(parsed['audio_binary'])
                    audio = audio_tensor.audio.numpy()
                    
                    # Record stats
                    test_results['sample_rates'].append(sample_rate)
                    test_results['clip_lengths'].append(len(audio))
                    clips_in_file += 1
                    
                    # Print first clip details
                    if clips_in_file == 1:
                        print(f"      Sample rate: {sample_rate}Hz")
                        print(f"      Clip shape: {audio.shape}")
                        print(f"      Source: {source_file} (clip {clip_index})")
                        print(f"      Dataset: {dataset_name}")
                        print(f"      Audio range: [{audio.min():.3f}, {audio.max():.3f}]")
                    
                except Exception as e:
                    test_results['errors'].append(f"Error parsing clip: {e}")
            
            print(f"      ✅ Successfully read {clips_in_file} clips")
            test_results['files_tested'] += 1
            test_results['clips_tested'] += clips_in_file
            
        except Exception as e:
            print(f"      ❌ Error reading file: {e}")
            test_results['errors'].append(f"Error reading {tfrecord_file.name}: {e}")
    
    # 4. Verify audio properties
    print("\n4. Verifying audio properties...")
    
    if test_results['sample_rates']:
        unique_rates = set(test_results['sample_rates'])
        if len(unique_rates) == 1:
            rate = list(unique_rates)[0]
            print(f"✅ Consistent sample rate: {rate}Hz")
            
            if rate == 44100:
                print("   ✓ Correct 44.1kHz sample rate")
            elif rate == 16000:
                print("   ⚠️  Using 16kHz (original rate) - update if you want 44.1kHz")
            else:
                print(f"   ⚠️  Unexpected sample rate: {rate}Hz")
        else:
            print(f"⚠️  Multiple sample rates found: {unique_rates}")
    
    if test_results['clip_lengths']:
        unique_lengths = set(test_results['clip_lengths'])
        expected_length = 44100 if 44100 in test_results['sample_rates'] else 16000
        
        if len(unique_lengths) == 1:
            length = list(unique_lengths)[0]
            print(f"✅ Consistent clip length: {length} samples")
            
            if length == expected_length:
                print(f"   ✓ Correct length for 1-second clips at {expected_length}Hz")
            else:
                print(f"   ⚠️  Unexpected length (expected {expected_length})")
        else:
            print(f"⚠️  Multiple clip lengths found: {unique_lengths}")
    
    # 5. Test data pipeline
    print("\n5. Testing training pipeline simulation...")
    
    try:
        # Create mini dataset
        test_dataset = tf.data.TFRecordDataset(
            [str(f) for f in tfrecord_files[:2]],
            compression_type='GZIP'
        )
        
        # Parse function
        def parse_tfrecord(example_proto):
            parsed = tf.io.parse_single_example(example_proto, feature_description)
            audio = tf.audio.decode_wav(parsed['audio_binary']).audio
            return tf.reshape(audio, [-1])
        
        # Build pipeline
        test_dataset = test_dataset.map(parse_tfrecord)
        test_dataset = test_dataset.batch(3)  # Simulate batching
        
        # Test mixing simulation
        for batch in test_dataset.take(1):
            mixed = tf.reduce_mean(batch, axis=0)
            print(f"✅ Pipeline test successful:")
            print(f"   - Batch shape: {batch.shape}")
            print(f"   - Mixed shape: {mixed.shape}")
            print(f"   - Mixed range: [{float(tf.reduce_min(mixed)):.3f}, {float(tf.reduce_max(mixed)):.3f}]")
    
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        test_results['errors'].append(f"Pipeline test: {e}")
    
    # 6. Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    print(f"\n✅ Files tested: {test_results['files_tested']}/{num_files_to_test}")
    print(f"✅ Clips tested: {test_results['clips_tested']}")
    
    if test_results['errors']:
        print(f"\n⚠️  Errors encountered ({len(test_results['errors'])}):")
        for error in test_results['errors'][:5]:  # Show first 5 errors
            print(f"   - {error}")
    else:
        print("\n🎉 All tests passed successfully!")
    
    return len(test_results['errors']) == 0


def test_audio_playback(tfrecord_dir, output_dir="./test_outputs"):
    """
    Extract and save a few audio clips for manual verification.
    
    Args:
        tfrecord_dir: Directory containing TFRecord files
        output_dir: Directory to save test audio files
    """
    print("\n" + "="*60)
    print("EXTRACTING SAMPLE AUDIO FOR PLAYBACK")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get first TFRecord file
    tfrecord_files = list(Path(tfrecord_dir).glob("*.tfrecord"))
    if not tfrecord_files:
        print("❌ No TFRecord files found")
        return
    
    tfrecord_file = tfrecord_files[0]
    print(f"\nExtracting clips from {tfrecord_file.name}")
    
    # Feature description
    feature_description = {
        'audio_binary': tf.io.FixedLenFeature([], tf.string),
        'sample_rate': tf.io.FixedLenFeature([], tf.int64),
        'source_file': tf.io.FixedLenFeature([], tf.string),
        'clip_index': tf.io.FixedLenFeature([], tf.int64),
        'dataset': tf.io.FixedLenFeature([], tf.string),
    }
    
    # Create dataset
    dataset = tf.data.TFRecordDataset(str(tfrecord_file), compression_type='GZIP')
    
    # Extract first 3 clips
    clips_extracted = 0
    for raw_record in dataset.take(3):
        parsed = tf.io.parse_single_example(raw_record, feature_description)
        
        # Get metadata
        source_file = parsed['source_file'].numpy().decode('utf-8')
        clip_index = parsed['clip_index'].numpy()
        sample_rate = parsed['sample_rate'].numpy()
        
        # Decode audio
        audio_binary = parsed['audio_binary'].numpy()
        
        # Save as WAV
        output_path = Path(output_dir) / f"test_{clips_extracted}_{source_file}_clip{clip_index}.wav"
        
        # Write the binary directly (it's already WAV format)
        with open(output_path, 'wb') as f:
            f.write(audio_binary)
        
        print(f"✅ Saved: {output_path.name} ({sample_rate}Hz)")
        clips_extracted += 1
    
    print(f"\n📁 Test audio files saved to: {output_dir}")
    print("   You can play these files to verify audio quality")


def test_gcs_compatibility(tfrecord_dir):
    """
    Test that TFRecords are ready for GCS upload.
    """
    print("\n" + "="*60)
    print("GCS UPLOAD READINESS CHECK")
    print("="*60)
    
    tfrecord_files = list(Path(tfrecord_dir).glob("*.tfrecord"))
    
    if not tfrecord_files:
        print("❌ No TFRecord files found")
        return False
    
    # Check file sizes
    total_size = 0
    file_sizes = []
    
    for f in tfrecord_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        file_sizes.append(size_mb)
        total_size += size_mb
    
    print(f"\n✅ Storage Statistics:")
    print(f"   - Number of files: {len(tfrecord_files)}")
    print(f"   - Total size: {total_size:.1f} MB ({total_size/1024:.2f} GB)")
    print(f"   - Average file size: {np.mean(file_sizes):.1f} MB")
    print(f"   - Size range: {min(file_sizes):.1f} - {max(file_sizes):.1f} MB")
    
    # Estimate upload time
    upload_speed_mbps = 10  # Assume 10 Mbps upload
    estimated_time = (total_size * 8) / (upload_speed_mbps * 60)
    print(f"\n📤 Upload estimate (at {upload_speed_mbps} Mbps): ~{estimated_time:.1f} minutes")
    
    # Generate upload command
    print(f"\n📋 GCS Upload Command:")
    print(f"   gsutil -m cp -r {tfrecord_dir}/*.tfrecord gs://your-bucket/tfrecords/")
    
    print(f"\n💡 Tips:")
    print(f"   - Use 'gsutil -m' for parallel uploads (up to 10x faster)")
    print(f"   - Consider using 'gsutil rsync' for resumable uploads")
    print(f"   - Monitor with: gsutil -m cp -r -P ...")
    
    return True


def run_all_tests(tfrecord_dir, output_dir="./test_outputs"):
    """
    Run all verification tests.
    
    Args:
        tfrecord_dir: Directory containing TFRecord files
        output_dir: Directory for test outputs
    """
    print("\n🔍 RUNNING COMPLETE VERIFICATION SUITE\n")
    
    # Run tests
    success = test_tfrecords(tfrecord_dir)
    test_audio_playback(tfrecord_dir, output_dir)
    test_gcs_compatibility(tfrecord_dir)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL VERIFICATION STATUS")
    print("="*60)
    
    if success:
        print("\n✅ ALL TESTS PASSED!")
        print("Your TFRecords are ready for training.")
        print("\nNext steps:")
        print("1. Upload to GCS using the command above")
        print("2. Test in Colab with a small training run")
        print("3. Start full training")
    else:
        print("\n⚠️  Some issues detected. Please review the errors above.")
        print("Common fixes:")
        print("- Check input data paths")
        print("- Verify audio file formats")
        print("- Ensure enough disk space")
    
    return success


# Quick test for Colab
def quick_colab_test(tfrecord_path):
    """
    Quick test to run in Colab after downloading from GCS.
    
    Usage in Colab:
        from test_preprocessing import quick_colab_test
        quick_colab_test("/content/data/tfrecords")
    """
    print("🚀 Quick Colab Test")
    
    # Check files exist
    files = tf.io.gfile.glob(f"{tfrecord_path}/*.tfrecord")
    print(f"Found {len(files)} TFRecord files")
    
    if not files:
        print("❌ No files found!")
        return
    
    # Test loading
    dataset = tf.data.TFRecordDataset(files[:5], compression_type='GZIP')
    
    count = 0
    for _ in dataset.take(10):
        count += 1
    
    print(f"✅ Successfully loaded {count} records")
    print("Ready for training!")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        tfrecord_directory = sys.argv[1]
    else:
        tfrecord_directory = "./tfrecords_44k"
    
    print(f"Testing TFRecords in: {tfrecord_directory}")
    run_all_tests(tfrecord_directory)