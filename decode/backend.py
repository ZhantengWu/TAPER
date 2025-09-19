"""
Decoder backend interfaces for QR code robust reading project.
Provides unified interface for different decoding libraries (pyzbar, ZXing).
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional
import sys
import os
from pathlib import Path

# Suppress zbar assertion warnings at the C library level
os.environ['ZBAR_VERBOSITY'] = '0'

try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False
    print("Warning: pyzbar not available. Install with: pip install pyzbar")


def safe_pyzbar_decode(img: np.ndarray) -> List:
    """
    Safe wrapper for pyzbar decoding that handles zbar assertion errors.
    
    Args:
        img: Input image (RGB or grayscale)
        
    Returns:
        List of decoded barcodes, empty if failed
    """
    import contextlib
    import sys
    
    # More aggressive stderr suppression for C library assertions
    original_stderr = sys.stderr
    try:
        sys.stderr = open(os.devnull, 'w')
        
        original_stderr_fd = os.dup(2)
        try:
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull_fd, 2)
        except:
            devnull_fd = None
        
        try:
            return pyzbar.decode(img)
        except Exception as e:
            error_msg = str(e).lower()
            if "zbar" in error_msg and "assertion" in error_msg:
                try:
                    if len(img.shape) == 3:
                        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = img
                    
                    smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
                    
                    return pyzbar.decode(smoothed)
                except:
                    return []
            else:
                raise e
        finally:
            os.dup2(original_stderr_fd, 2)
            if devnull_fd is not None:
                os.close(devnull_fd)
            os.close(original_stderr_fd)
    finally:
        if sys.stderr != original_stderr:
            sys.stderr.close()
        sys.stderr = original_stderr


class DecoderBackend(ABC):
    """Abstract base class for decoder backends."""
    
    @abstractmethod
    def decode(self, img_bgr: np.ndarray) -> List[str]:
        """
        Decode QR codes/barcodes from image.
        
        Args:
            img_bgr: Input image in BGR format (OpenCV format)
            
        Returns:
            List of decoded text strings (empty if no codes found)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get backend name."""
        pass


class PyzbarBackend(DecoderBackend):
    """Pyzbar decoder backend."""
    
    def __init__(self):
        if not PYZBAR_AVAILABLE:
            raise ImportError("pyzbar is not available. Install with: pip install pyzbar")
    
    def decode(self, img_bgr: np.ndarray) -> List[str]:
        """
        Decode using pyzbar library.
        
        Args:
            img_bgr: Input image in BGR format
            
        Returns:
            List of decoded text strings
        """
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            decoded_objects = safe_pyzbar_decode(img_rgb)
            
            texts = []
            for obj in decoded_objects:
                try:
                    text = obj.data.decode('utf-8')
                except UnicodeDecodeError:
                    text = obj.data.decode('latin-1')
                texts.append(text)
            
            return texts
            
        except Exception as e:
            return []
    
    @property
    def name(self) -> str:
        return "pyzbar"


class QReaderBackend(DecoderBackend):
    """QReader decoder backend using qreader library."""
    
    def __init__(self):
        """Initialize QReader backend."""
        self._check_qreader_availability()
    
    def _check_qreader_availability(self) -> None:
        """Check if qreader is available."""
        try:
            from qreader import QReader
            self.qreader = QReader()
            print(f"QReader initialized successfully")
        except ImportError:
            print("Warning: qreader not found. Install with: pip install qreader")
            self.qreader = None
        except Exception as e:
            print(f"Warning: QReader initialization failed: {e}")
            self.qreader = None
    
    def decode(self, img_bgr: np.ndarray) -> List[str]:
        """
        Decode using qreader library.
        
        Args:
            img_bgr: Input image in BGR format
            
        Returns:
            List of decoded text strings
        """
        if self.qreader is None:
            print("QReader not available")
            return []
        
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            print(f"Image shape: {img_rgb.shape}, dtype: {img_rgb.dtype}")
            
            try:
                result = self.qreader.detect_and_decode(image=img_rgb)
                print(f"detect_and_decode result: {result}, type: {type(result)}")
            except Exception as e1:
                print(f"detect_and_decode failed: {e1}")
                try:
                    detections = self.qreader.detect(image=img_rgb)
                    print(f"Detections: {detections}")
                    if detections:
                        result = self.qreader.decode(image=img_rgb, detection_result=detections)
                        print(f"decode result: {result}")
                    else:
                        result = None
                except Exception as e2:
                    print(f"separate detect/decode failed: {e2}")
                    result = None
            
            print(f"QReader result: {result}, type: {type(result)}")
            
            if result is None or result == "" or result == ():
                print("No QR codes detected")
                return []
            elif isinstance(result, str) and result.strip():
                print(f"Single QR code decoded: {result}")
                return [result.strip()]
            elif isinstance(result, (list, tuple)) and len(result) > 0:
                texts = []
                for i, r in enumerate(result):
                    print(f"Result {i}: {r}, type: {type(r)}")
                    if r is not None and isinstance(r, str) and r.strip():
                        texts.append(r.strip())
                print(f"QR codes decoded: {texts}")
                return texts
            else:
                print(f"Unexpected result type: {type(result)}, value: {result}")
                return []
            
        except Exception as e:
            print(f"QReader decode error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    @property
    def name(self) -> str:
        return "qreader"


class DBRBackend(DecoderBackend):
    """Dynamsoft Barcode Reader backend using dbr library."""
    
    def __init__(self):
        """Initialize DBR backend."""
        self._check_dbr_availability()
    
    def _check_dbr_availability(self) -> None:
        """Check if dbr is available."""
        try:
            from dbr import BarcodeReader
            self.reader = BarcodeReader()
        except ImportError:
            print("Warning: dbr not found. Install with: pip install dbr")
            self.reader = None
        except Exception as e:
            print(f"Warning: DBR initialization failed: {e}")
            self.reader = None
    
    def decode(self, img_bgr: np.ndarray) -> List[str]:
        """
        Decode using Dynamsoft Barcode Reader.
        
        Args:
            img_bgr: Input image in BGR format
            
        Returns:
            List of decoded text strings
        """
        if self.reader is None:
            return []
        
        try:
            text_results = self.reader.decode_buffer(img_bgr)
            
            texts = []
            if text_results is not None:
                for result in text_results:
                    if hasattr(result, 'barcode_text') and result.barcode_text:
                        text = result.barcode_text.strip()
                        if len(text.strip()) > 0:
                            texts.append(text)
            
            return texts
            
        except Exception:
            return []
    
    @property
    def name(self) -> str:
        return "dbr"


class ZXingBackend(DecoderBackend):
    """ZXing decoder backend using zxing-cpp Python library."""
    
    def __init__(self):
        """Initialize ZXing backend."""
        self._check_zxing_availability()
    
    def _check_zxing_availability(self) -> None:
        """Check if zxing-cpp is available."""
        try:
            import zxingcpp
            self.zxing = zxingcpp
        except ImportError:
            print("Warning: zxing-cpp not found. Install with: pip install zxing-cpp")
            self.zxing = None
    
    def decode(self, img_bgr: np.ndarray) -> List[str]:
        """
        Decode using zxing-cpp Python library.
        
        Args:
            img_bgr: Input image in BGR format
            
        Returns:
            List of decoded text strings
        """
        if self.zxing is None:
            return []
        
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            results = self.zxing.read_barcodes(img_rgb)
            
            texts = []
            for result in results:
                if result.valid and result.text:
                    texts.append(result.text)
            
            return texts
            
        except Exception as e:
            print(f"ZXing decoding error: {e}")
            return []
    
    @property
    def name(self) -> str:
        return "zxing"


def create_backend(backend_name: str = "pyzbar", **kwargs) -> DecoderBackend:
    """
    Create decoder backend by name.
    
    Args:
        backend_name: Backend name ("pyzbar", "zxing", "qreader", or "dbr")
        **kwargs: Additional arguments for backend initialization
        
    Returns:
        Decoder backend instance
    """
    if backend_name.lower() == "pyzbar":
        return PyzbarBackend()
    elif backend_name.lower() == "zxing":
        return ZXingBackend()
    elif backend_name.lower() == "qreader":
        return QReaderBackend()
    elif backend_name.lower() == "dbr":
        return DBRBackend()
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

def get_default_backend() -> DecoderBackend:
    """Get default decoder backend (ZXing)."""
    return ZXingBackend()
