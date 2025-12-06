"""
Universal audio converter that handles all formats and converts to WAV
"""

import os
import tempfile
import logging
import subprocess
import shutil
from typing import Optional, Tuple, Dict, Any, List
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)


class UniversalAudioConverter:
    """
    Universal audio converter that can handle any format and convert to WAV
    """
    
    def __init__(self, keep_converted_files: bool = False):
        """
        Args:
            keep_converted_files: Whether to keep converted files for debugging
        """
        self.keep_converted_files = keep_converted_files
        self.ffmpeg_available = self._check_ffmpeg()
        self.pydub_available = self._check_pydub()
        self.librosa_available = self._check_librosa()
        self.soundfile_available = self._check_soundfile()
        
        # Create output directory with ABSOLUTE PATH
        self.output_dir = Path("data/converted_audio").absolute()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Converter initialized. Output directory: {self.output_dir}")
        logger.info(f"Converter status - FFmpeg: {self.ffmpeg_available}, "
                   f"Pydub: {self.pydub_available}, "
                   f"Librosa: {self.librosa_available}, "
                   f"Soundfile: {self.soundfile_available}")
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=2
            )
            return result.returncode == 0
        except:
            return False
    
    def _check_pydub(self) -> bool:
        """Check if pydub is available"""
        try:
            import pydub
            return True
        except ImportError:
            return False
    
    def _check_librosa(self) -> bool:
        """Check if librosa is available"""
        try:
            import librosa
            return True
        except ImportError:
            return False
    
    def _check_soundfile(self) -> bool:
        """Check if soundfile is available"""
        try:
            import soundfile
            return True
        except ImportError:
            return False
    
    def convert_to_wav(self, 
                      input_path: str,
                      output_path: Optional[str] = None,
                      sample_rate: int = 16000,
                      mono: bool = True,
                      keep_temp: bool = False) -> str:
        """
        Convert any audio file to WAV format
        
        Args:
            input_path: Path to input audio file (converted to absolute path)
            output_path: Path to output WAV file (creates temp file if None)
            sample_rate: Target sample rate
            mono: Convert to mono
            keep_temp: Keep temporary files for debugging
            
        Returns:
            str: ABSOLUTE path to converted WAV file
        """
        # Convert input path to absolute
        input_path = str(Path(input_path).absolute())
        
        # Validate input
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Create output path if not provided
        if output_path is None:
            # Create a descriptive filename in our output directory
            import time
            import uuid
            input_name = Path(input_path).stem
            timestamp = int(time.time())
            unique_id = uuid.uuid4().hex[:8]
            output_filename = f"{input_name}_{timestamp}_{unique_id}.wav"
            output_path = str(self.output_dir / output_filename)
        else:
            # Convert output path to absolute
            output_path = str(Path(output_path).absolute())
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get file extension for format detection
        file_ext = os.path.splitext(input_path)[1].lower()
        
        # Try conversion methods in priority order
        methods = [
            ("FFmpeg", self._convert_with_ffmpeg),
            ("Pydub", self._convert_with_pydub),
            ("Librosa", self._convert_with_librosa),
            ("Soundfile", self._convert_with_soundfile)
        ]
        
        last_error = None
        
        for method_name, method_func in methods:
            # Check if method is available and suitable for this format
            if not self._is_method_suitable(method_name, file_ext):
                continue
            
            try:
                logger.info(f"Attempting conversion with {method_name}...")
                result = method_func(input_path, output_path, sample_rate, mono)
                
                if result and os.path.exists(result):
                    file_size = os.path.getsize(result)
                    logger.info(f"Successfully converted with {method_name} to {result} ({file_size} bytes)")
                    
                    # Verify the file is readable
                    if self._verify_wav_file(result):
                        logger.info(f"WAV file verified and ready for transcription")
                    
                    return result
                    
            except Exception as e:
                last_error = e
                logger.warning(f"{method_name} conversion failed: {str(e)}")
                continue
        
        # If all methods failed
        error_msg = f"Failed to convert {input_path} to WAV. "
        if last_error:
            error_msg += f"Last error: {str(last_error)}"
        else:
            error_msg += "No suitable conversion method found."
        
        raise ValueError(error_msg)
    
    def _verify_wav_file(self, file_path: str) -> bool:
        """Verify that a WAV file is readable"""
        try:
            import wave
            with wave.open(file_path, 'rb') as wav_file:
                # Check basic properties
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                
                logger.debug(f"WAV file verified: {frames} frames, {rate}Hz, {channels} channels")
                return frames > 0 and rate > 0
        except Exception as e:
            logger.warning(f"WAV file verification failed for {file_path}: {e}")
            return False
    
    def _is_method_suitable(self, method_name: str, file_ext: str) -> bool:
        """Check if a conversion method is suitable for the file format"""
        if method_name == "FFmpeg":
            return self.ffmpeg_available
        elif method_name == "Pydub":
            return self.pydub_available and file_ext in ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.wma', '.aiff', '.opus']
        elif method_name == "Librosa":
            return self.librosa_available and file_ext in ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac', '.opus']
        elif method_name == "Soundfile":
            return self.soundfile_available and file_ext in ['.wav', '.flac', '.ogg', '.aiff']
        return False
    
    def _convert_with_ffmpeg(self, 
                           input_path: str, 
                           output_path: str,
                           sample_rate: int,
                           mono: bool) -> str:
        """Convert using FFmpeg"""
        # Ensure paths are quoted for Windows if they contain spaces
        import shlex
        input_quoted = shlex.quote(input_path) if ' ' in input_path else input_path
        output_quoted = shlex.quote(output_path) if ' ' in output_path else output_path
        
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i", input_quoted,
            "-ac", "1" if mono else "2",
            "-ar", str(sample_rate),
            "-acodec", "pcm_s16le",
            "-f", "wav",
            output_quoted
        ]
        
        logger.debug(f"Running FFmpeg command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        return output_path
    
    def _convert_with_pydub(self,
                          input_path: str,
                          output_path: str,
                          sample_rate: int,
                          mono: bool) -> str:
        """Convert using pydub"""
        from pydub import AudioSegment
        
        # Determine format from extension
        format_map = {
            '.mp3': 'mp3',
            '.wav': 'wav',
            '.ogg': 'ogg',
            '.flac': 'flac',
            '.aac': 'aac',
            '.m4a': 'mp4',
            '.wma': 'wma',
            '.aiff': 'aiff',
            '.opus': 'opus'
        }
        
        file_ext = os.path.splitext(input_path)[1].lower()
        format_str = format_map.get(file_ext, file_ext[1:] if file_ext else 'mp3')
        
        try:
            # Load audio
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio = AudioSegment.from_file(input_path, format=format_str)
            
            # Convert to mono if needed
            if mono and audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Set sample rate
            if audio.frame_rate != sample_rate:
                audio = audio.set_frame_rate(sample_rate)
            
            # Export to WAV
            audio.export(output_path, format="wav", parameters=["-ac", "1" if mono else "2"])
            
            return output_path
            
        except Exception as e:
            # If pydub fails with format detection, try without format
            try:
                audio = AudioSegment.from_file(input_path)
                if mono and audio.channels > 1:
                    audio = audio.set_channels(1)
                if audio.frame_rate != sample_rate:
                    audio = audio.set_frame_rate(sample_rate)
                audio.export(output_path, format="wav")
                return output_path
            except:
                raise e
    
    def _convert_with_librosa(self,
                            input_path: str,
                            output_path: str,
                            sample_rate: int,
                            mono: bool) -> str:
        """Convert using librosa"""
        import librosa
        import soundfile as sf
        
        # Load audio
        audio, orig_sr = librosa.load(
            input_path,
            sr=None,
            mono=mono
        )
        
        # Resample if needed
        if orig_sr != sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=orig_sr,
                target_sr=sample_rate
            )
        
        # Ensure audio is mono (librosa.load with mono=True should handle this)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Save as WAV
        sf.write(output_path, audio, sample_rate)
        
        return output_path
    
    def _convert_with_soundfile(self,
                              input_path: str,
                              output_path: str,
                              sample_rate: int,
                              mono: bool) -> str:
        """Convert using soundfile"""
        import soundfile as sf
        
        # Read audio
        audio, orig_sr = sf.read(input_path)
        
        # Convert to mono if needed
        if mono and len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Resample if needed
        if orig_sr != sample_rate:
            import librosa
            audio = librosa.resample(
                y=audio,
                orig_sr=orig_sr,
                target_sr=sample_rate
            )
        
        # Save as WAV
        sf.write(output_path, audio, sample_rate)
        
        return output_path
    
    def convert_bytes_to_wav(self,
                           audio_bytes: bytes,
                           original_filename: Optional[str] = None,
                           output_path: Optional[str] = None,
                           sample_rate: int = 16000,
                           mono: bool = True,
                           keep_temp: bool = False) -> Tuple[str, List[str]]:
        """
        Convert audio bytes to WAV file
        
        Returns:
            Tuple[str, List[str]]: (ABSOLUTE wav_file_path, temp_files_to_clean)
        """
        temp_files = []
        
        try:
            # Create a temp directory in our output folder
            temp_dir = (self.output_dir / "temp").absolute()
            temp_dir.mkdir(exist_ok=True)
            
            # Create temp input file with original extension if available
            if original_filename:
                file_ext = os.path.splitext(original_filename)[1].lower()
                if not file_ext or file_ext == '.':
                    file_ext = '.audio'
            else:
                file_ext = '.audio'
            
            # Create unique temp file name
            import time
            import uuid
            temp_name = f"input_{int(time.time())}_{uuid.uuid4().hex[:8]}{file_ext}"
            input_temp_path = str(temp_dir / temp_name)
            
            with open(input_temp_path, 'wb') as f:
                f.write(audio_bytes)
            
            # Only add to cleanup list if we're not keeping temps
            if not keep_temp:
                temp_files.append(input_temp_path)
            
            # Create output path if not provided
            if output_path is None:
                output_name = f"converted_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"
                output_path = str(self.output_dir / output_name)
            else:
                output_path = str(Path(output_path).absolute())
            
            # Convert to WAV
            wav_path = self.convert_to_wav(
                input_temp_path,
                output_path,
                sample_rate=sample_rate,
                mono=mono,
                keep_temp=keep_temp
            )
            
            # If we're keeping the file, don't add it to cleanup list
            if not keep_temp:
                temp_files.append(wav_path)
            
            logger.info(f"Conversion complete. Input: {len(audio_bytes)} bytes, Output: {wav_path}")
            
            return wav_path, temp_files
            
        except Exception as e:
            # Clean up on error
            self._cleanup_files(temp_files)
            raise
    
    def _cleanup_files(self, file_paths: List[str]):
        """Clean up temporary files"""
        for path in file_paths:
            try:
                if os.path.exists(path):
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        os.unlink(path)
                        logger.debug(f"Cleaned up temp file: {path}")
            except Exception as e:
                logger.warning(f"Failed to clean up {path}: {e}")


# Import time for timestamp
import time

# Global instance - keep converted files for debugging during development
audio_converter = UniversalAudioConverter(keep_converted_files=True)