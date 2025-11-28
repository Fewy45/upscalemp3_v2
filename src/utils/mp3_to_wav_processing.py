"""
Preprocessing pipeline for Medley 2.0 and MUSDB18 datasets at 44.1kHz/16-bit.
This script processes mixture audio files and creates TFRecords for training.

WORKFLOW EXPLANATION:
1. LOCAL: Process audio files → Create TFRecords (compressed, efficient format)
2. UPLOAD: Transfer TFRecords to Google Cloud Storage (GCS) 
3. COLAB: Download TFRecords from GCS → Train model with streaming data

WHY THIS WORKFLOW?
- LOCAL PREPROCESSING: Heavy audio processing is done once locally (faster, no Colab timeouts)
- GCS STORAGE: Persistent storage that survives Colab disconnections
- TFRECORDS: Efficient binary format that streams data during training (no memory limits)
- COLAB TRAINING: Free GPU access for model training without preprocessing overhead
"""

import os
import numpy as np
import soundfile as sf
import librosa
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.signal import windows
import json
import hashlib
import wave
import io

class MedleyMusdbProcessor:
    """
    Audio processor for Medley 2.0 and MUSDB18 datasets at 44.1kHz/16-bit.
    """
    def __init__(self, clip_duration_seconds=1.0, window_overlap_ratio=0.25):
        self.clip_duration_seconds = clip_duration_seconds
        self.window_overlap_ratio = window_overlap_ratio
        self.sample_rate = 44100  # 44.1kHz instead of 16kHz
        self.bit_depth = 16
        self.samples_per_clip = int(clip_duration_seconds * self.sample_rate)
        self.step_size = int(self.samples_per_clip * (1 - window_overlap_ratio))
        
    def find_mixture_files(self, base_dir, dataset_type="auto"):
        """
        Find all mixture audio files in Medley 2.0 or MUSDB18 structure.
        
        Args:
            base_dir: Base directory containing the dataset
            dataset_type: "medley", "musdb", or "auto" (auto-detect)
        
        Returns:
            List of paths to mixture audio files
        """
        mixture_files = []
        base_path = Path(base_dir)
        print(f"Searching for mixture files in: {base_path}")
        
        # Auto-detect dataset type
        if dataset_type == "auto":
            if (base_path / "V2").exists():
                dataset_type = "medley"
            elif (base_path / "train").exists() or (base_path / "test").exists():
                dataset_type = "musdb"
            else:
                # Try both patterns
                print("Auto-detection failed, trying both patterns...")
        
        # Medley 2.0 pattern: Downloads/V2/<Artist_Song>/Artist_Song_MIX.wav
        if dataset_type in ["medley", "auto"]:
            medley_pattern = base_path / "V2" / "*" / "*_MIX.wav"
            medley_files = list(base_path.glob("V2/*/*_MIX.wav"))
            if medley_files:
                print(f"Found {len(medley_files)} Medley 2.0 mixture files")
                mixture_files.extend(medley_files)
        
        # MUSDB18 pattern: Downloads/MUSDB18_HQ/train/<Artist - Song>/mixture.wav
        #                   Downloads/MUSDB18_HQ/test/<Artist - Song>/mixture.wav
        if dataset_type in ["musdb", "auto"]:
            musdb_train = list(base_path.glob("MUSDB18_HQ/train/*/mixture.wav"))
            musdb_test = list(base_path.glob("MUSDB18_HQ/test/*/mixture.wav"))
            musdb_files = musdb_train + musdb_test
            if musdb_files:
                print(f"Found {len(musdb_train)} MUSDB18 train and {len(musdb_test)} test mixture files")
                mixture_files.extend(musdb_files)
        
        print(f"Total mixture files found: {len(mixture_files)}")
        return mixture_files
    
    def load_and_normalize_audio(self, file_path):
        """
        Load audio at 44.1kHz and normalize to 16-bit range.
        """
        try:
            # Load audio at original sample rate
            audio, sr = sf.read(str(file_path))
            
            # Resample to 44.1kHz if necessary
            if sr != self.sample_rate:
                print(f"Resampling {file_path.name} from {sr}Hz to {self.sample_rate}Hz")
                # Handle mono or stereo
                if len(audio.shape) == 1:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                else:
                    # Resample each channel
                    audio = np.array([
                        librosa.resample(audio[:, i], orig_sr=sr, target_sr=self.sample_rate)
                        for i in range(audio.shape[1])
                    ]).T
            
            # Convert stereo to mono by averaging channels
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Normalize to [-1, 1] range for processing
            audio = audio.astype(np.float32)
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            
            return audio
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def split_into_clips(self, audio):
        """
        Split audio into overlapping clips with windowing.
        """
        if len(audio) < self.samples_per_clip:
            # Pad short audio
            audio = np.pad(audio, (0, self.samples_per_clip - len(audio)))
        
        # Calculate number of clips
        num_clips = max(1, (len(audio) - self.samples_per_clip) // self.step_size + 1)
        
        # Create Hann window for smooth transitions
        window = windows.hann(self.samples_per_clip)
        # Apply lighter windowing (75% rectangular + 25% Hann)
        window = 0.25 * window + 0.75
        
        clips = []
        for i in range(num_clips):
            start = i * self.step_size
            end = start + self.samples_per_clip
            
            if end <= len(audio):
                clip = audio[start:end] * window
            else:
                # Handle last clip with padding
                remaining = audio[start:]
                clip = np.pad(remaining, (0, self.samples_per_clip - len(remaining)))
                clip = clip * window
            
            clips.append(clip)
        
        return clips
    
    def audio_to_wav_bytes(self, audio_clip):
        """
        Convert audio numpy array to WAV format bytes at 44.1kHz/16-bit.
        """
        wav_buffer = io.BytesIO()
        
        # Convert to 16-bit integer range
        audio_int16 = np.clip(audio_clip * 32767, -32768, 32767).astype(np.int16)
        
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)  # 44.1kHz
            wav_file.writeframes(audio_int16.tobytes())
        
        return wav_buffer.getvalue()


class TFRecordCreator:
    """
    Creates TFRecords from processed audio clips.
    """
    def __init__(self, processor):
        self.processor = processor
    
    def create_tf_example(self, audio_clip, metadata):
        """
        Create a TensorFlow Example from an audio clip.
        """
        # Convert audio to WAV bytes
        audio_bytes = self.processor.audio_to_wav_bytes(audio_clip)
        
        # Create feature dict
        feature = {
            'audio_binary': tf.train.Feature(bytes_list=tf.train.BytesList(value=[audio_bytes])),
            'sample_rate': tf.train.Feature(int64_list=tf.train.Int64List(value=[self.processor.sample_rate])),
            'samples': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(audio_clip)])),
            'source_file': tf.train.Feature(bytes_list=tf.train.BytesList(value=[metadata['source_file'].encode()])),
            'clip_index': tf.train.Feature(int64_list=tf.train.Int64List(value=[metadata['clip_index']])),
            'dataset': tf.train.Feature(bytes_list=tf.train.BytesList(value=[metadata['dataset'].encode()])),
        }
        
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()
    
    def process_file_to_examples(self, audio_path, dataset_name):
        """
        Process a single audio file into TF Examples.
        """
        audio = self.processor.load_and_normalize_audio(audio_path)
        if audio is None:
            return []
        
        clips = self.processor.split_into_clips(audio)
        examples = []
        
        for i, clip in enumerate(clips):
            metadata = {
                'source_file': audio_path.name,
                'clip_index': i,
                'dataset': dataset_name,
            }
            example = self.create_tf_example(clip, metadata)
            examples.append(example)
        
        return examples


def create_tfrecords_from_datasets(
    medley_dir=None,
    musdb_dir=None,
    output_dir="./tfrecords",
    clips_per_tfrecord=500,
    num_shards=500  # Distribute across 500 files as per original code
):
    """
    Main function to process Medley 2.0 and/or MUSDB18 datasets into TFRecords.
    
    This function:
    1. Finds all mixture files from the datasets
    2. Processes them into 1-second clips at 44.1kHz/16-bit
    3. Creates sharded TFRecords for efficient training
    4. Distributes clips evenly across shards using hash-based assignment
    
    Args:
        medley_dir: Path to Medley 2.0 dataset (contains V2 folder)
        musdb_dir: Path to MUSDB18 dataset (contains MUSDB18_HQ folder)
        output_dir: Output directory for TFRecord files
        clips_per_tfrecord: Approximate clips per TFRecord file
        num_shards: Number of TFRecord shards to create
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize processor
    processor = MedleyMusdbProcessor(
        clip_duration_seconds=1.0,
        window_overlap_ratio=0.25  # 25% overlap for better reconstruction
    )
    creator = TFRecordCreator(processor)
    
    # Find all mixture files
    all_mixture_files = []
    dataset_labels = []
    
    if medley_dir:
        medley_files = processor.find_mixture_files(medley_dir, "medley")
        all_mixture_files.extend(medley_files)
        dataset_labels.extend(["medley"] * len(medley_files))
    
    if musdb_dir:
        musdb_files = processor.find_mixture_files(musdb_dir, "musdb")
        all_mixture_files.extend(musdb_files)
        dataset_labels.extend(["musdb"] * len(musdb_files))
    
    if not all_mixture_files:
        print("No mixture files found! Check your dataset paths.")
        return
    
    print(f"\nProcessing {len(all_mixture_files)} total mixture files")
    print(f"Output directory: {output_dir}")
    print(f"Creating {num_shards} TFRecord shards")
    
    # Create shard writers
    shard_writers = {}
    shard_counts = {}
    options = tf.io.TFRecordOptions(compression_type='GZIP', compression_level=6)
    
    for i in range(num_shards):
        shard_path = os.path.join(output_dir, f"{i:03d}.tfrecord")
        shard_writers[i] = tf.io.TFRecordWriter(shard_path, options=options)
        shard_counts[i] = 0
    
    # Process files with progress bar
    total_clips = 0
    for audio_path, dataset_name in tqdm(
        zip(all_mixture_files, dataset_labels),
        total=len(all_mixture_files),
        desc="Processing audio files"
    ):
        examples = creator.process_file_to_examples(audio_path, dataset_name)
        
        # Distribute examples across shards using hash
        for example in examples:
            # Use hash to determine shard (similar to original code)
            shard_idx = hash(example) % num_shards
            shard_writers[shard_idx].write(example)
            shard_counts[shard_idx] += 1
            total_clips += 1
    
    # Close all writers
    for writer in shard_writers.values():
        writer.close()
    
    # Print statistics
    print(f"\n{'='*50}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"Total clips generated: {total_clips}")
    print(f"Average clips per shard: {total_clips / num_shards:.1f}")
    print(f"Clips per shard range: {min(shard_counts.values())} - {max(shard_counts.values())}")
    
    # Save metadata
    metadata = {
        'total_clips': total_clips,
        'num_shards': num_shards,
        'sample_rate': processor.sample_rate,
        'clip_duration': processor.clip_duration_seconds,
        'overlap_ratio': processor.window_overlap_ratio,
        'datasets': {
            'medley': len([d for d in dataset_labels if d == 'medley']),
            'musdb': len([d for d in dataset_labels if d == 'musdb'])
        }
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to: {metadata_path}")
    
    # Print next steps
    print(f"\n{'='*50}")
    print("NEXT STEPS:")
    print(f"{'='*50}")
    print("1. Upload TFRecords to Google Cloud Storage:")
    print(f"   gsutil -m cp -r {output_dir}/*.tfrecord gs://parrotfish-audio-data/tfrecords_44k/")
    print("\n2. In Colab, download the TFRecords:")
    print("   !mkdir -p /content/data")
    print("   !gsutil -m cp -r gs://parrotfish-audio-data/tfrecords_44k /content/data/")
    print("\n3. Train your model with the TFRecords:")
    print("   train_model(tfrecords_dir='/content/data/tfrecords_44k', ...)")
    
    return total_clips


# Example usage
if __name__ == "__main__":
    # Set your dataset paths here
    MEDLEY_PATH = "data"  # Should contain V2 folder
    MUSDB_PATH = "data"  # Should contain MUSDB18_HQ folder
    OUTPUT_PATH = "./tfrecords_44k"
    
    # Process both datasets
    create_tfrecords_from_datasets(
        medley_dir=MEDLEY_PATH,
        musdb_dir=MUSDB_PATH,
        output_dir=OUTPUT_PATH,
        num_shards=500  # Creates 500 TFRecord files as per original design
    )