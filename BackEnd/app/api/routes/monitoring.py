# app/api/routes/monitoring.py
"""
Monitoring and latency tracking endpoints
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import Dict, List, Optional
import logging
import asyncio
import time
from datetime import datetime, timedelta
from collections import deque
import statistics

router = APIRouter()
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor system performance metrics"""
    
    def __init__(self, max_samples: int = 1000):
        self.metrics = {
            "transcription": {
                "latencies": deque(maxlen=max_samples),
                "success_count": 0,
                "error_count": 0,
                "last_timestamp": None
            },
            "speaker_id": {
                "latencies": deque(maxlen=max_samples),
                "success_count": 0,
                "error_count": 0,
                "accuracy": deque(maxlen=max_samples),
                "last_timestamp": None
            },
            "tts": {
                "latencies": deque(maxlen=max_samples),
                "success_count": 0,
                "error_count": 0,
                "last_timestamp": None
            },
            "wake_word": {
                "detection_count": 0,
                "false_positives": 0,
                "latencies": deque(maxlen=max_samples),
                "last_timestamp": None
            },
            "websocket": {
                "connections": 0,
                "messages_sent": 0,
                "messages_received": 0,
                "connection_durations": deque(maxlen=max_samples)
            }
        }
        
        self.active_requests = {}
        self.system_start_time = datetime.now()
    
    def start_request(self, request_id: str, request_type: str):
        """Start timing a request"""
        self.active_requests[request_id] = {
            "type": request_type,
            "start_time": time.time(),
            "timestamp": datetime.now()
        }
    
    def end_request(self, request_id: str, success: bool = True, 
                   accuracy: float = None, metadata: dict = None):
        """End timing a request and record metrics"""
        if request_id not in self.active_requests:
            return
        
        request_data = self.active_requests.pop(request_id)
        request_type = request_data["type"]
        latency = time.time() - request_data["start_time"]
        
        if request_type in self.metrics:
            metric = self.metrics[request_type]
            metric["latencies"].append(latency)
            metric["last_timestamp"] = datetime.now()
            
            if success:
                metric["success_count"] += 1
            else:
                metric["error_count"] += 1
            
            if accuracy is not None:
                metric["accuracy"].append(accuracy)
    
    def record_wake_word_detection(self, is_correct: bool = True, latency: float = None):
        """Record wake word detection"""
        metric = self.metrics["wake_word"]
        metric["detection_count"] += 1
        metric["last_timestamp"] = datetime.now()
        
        if not is_correct:
            metric["false_positives"] += 1
        
        if latency is not None:
            metric["latencies"].append(latency)
    
    def update_websocket_stats(self, connections: int = 0, 
                             sent: int = 0, received: int = 0,
                             connection_duration: float = None):
        """Update WebSocket statistics"""
        metric = self.metrics["websocket"]
        metric["connections"] = connections
        metric["messages_sent"] += sent
        metric["messages_received"] += received
        
        if connection_duration is not None:
            metric["connection_durations"].append(connection_duration)
    
    def get_metrics(self, timeframe_minutes: int = 5) -> Dict:
        """Get aggregated metrics for specified timeframe"""
        cutoff_time = datetime.now() - timedelta(minutes=timeframe_minutes)
        
        result = {
            "system": {
                "uptime_seconds": (datetime.now() - self.system_start_time).total_seconds(),
                "total_requests": sum(m["success_count"] + m["error_count"] 
                                    for m in self.metrics.values() 
                                    if "success_count" in m),
                "active_requests": len(self.active_requests)
            },
            "services": {}
        }
        
        for service_name, metrics in self.metrics.items():
            # Filter recent latencies
            recent_latencies = list(metrics.get("latencies", []))[-100:]  # Last 100 samples
            
            service_metrics = {
                "status": "healthy" if metrics.get("error_count", 0) == 0 else "degraded",
                "total_requests": metrics.get("success_count", 0) + metrics.get("error_count", 0),
                "success_rate": self._calculate_success_rate(metrics),
                "latency_ms": self._calculate_latency_stats(recent_latencies),
                "last_activity": metrics.get("last_timestamp"),
                "active": metrics.get("last_timestamp") is not None and 
                         (datetime.now() - metrics.get("last_timestamp")).total_seconds() < 300
            }
            
            # Add service-specific metrics
            if "accuracy" in metrics and metrics["accuracy"]:
                service_metrics["accuracy"] = {
                    "mean": statistics.mean(metrics["accuracy"]),
                    "min": min(metrics["accuracy"]),
                    "max": max(metrics["accuracy"]),
                    "samples": len(metrics["accuracy"])
                }
            
            if service_name == "wake_word":
                service_metrics.update({
                    "detection_count": metrics.get("detection_count", 0),
                    "false_positive_rate": (
                        metrics.get("false_positives", 0) / 
                        max(1, metrics.get("detection_count", 1))
                    )
                })
            
            if service_name == "websocket":
                service_metrics.update({
                    "active_connections": metrics.get("connections", 0),
                    "total_messages": metrics.get("messages_sent", 0) + 
                                    metrics.get("messages_received", 0),
                    "avg_connection_duration": (
                        statistics.mean(metrics["connection_durations"]) 
                        if metrics["connection_durations"] else 0
                    )
                })
            
            result["services"][service_name] = service_metrics
        
        return result
    
    def _calculate_success_rate(self, metrics: Dict) -> float:
        """Calculate success rate from metrics"""
        success = metrics.get("success_count", 0)
        errors = metrics.get("error_count", 0)
        total = success + errors
        return (success / total * 100) if total > 0 else 100.0
    
    def _calculate_latency_stats(self, latencies: List[float]) -> Dict:
        """Calculate latency statistics"""
        if not latencies:
            return {"mean": 0, "min": 0, "max": 0, "p95": 0, "samples": 0}
        
        return {
            "mean_ms": statistics.mean(latencies) * 1000,
            "min_ms": min(latencies) * 1000,
            "max_ms": max(latencies) * 1000,
            "p95_ms": statistics.quantiles(latencies, n=20)[18] * 1000 
                     if len(latencies) >= 20 else max(latencies) * 1000,
            "samples": len(latencies)
        }


# Global monitor instance
monitor = PerformanceMonitor()


@router.websocket("/ws/monitoring")
async def websocket_monitoring_endpoint(websocket: WebSocket):
    """WebSocket for real-time monitoring data"""
    await websocket.accept()
    
    try:
        while True:
            # Get current metrics
            metrics = monitor.get_metrics()
            
            # Add timestamp
            metrics["timestamp"] = datetime.now().isoformat()
            metrics["server_time"] = time.time()
            
            # Send metrics
            await websocket.send_json(metrics)
            
            # Update WebSocket stats
            monitor.update_websocket_stats(sent=1)
            
            # Wait before next update
            await asyncio.sleep(1)  # Update every second
    
    except WebSocketDisconnect:
        logger.info("Monitoring WebSocket disconnected")
    except Exception as e:
        logger.error(f"Monitoring WebSocket error: {e}")


@router.get("/metrics")
async def get_metrics(
    timeframe_minutes: int = Query(5, ge=1, le=60),
    service: Optional[str] = Query(None)
):
    """Get current performance metrics"""
    metrics = monitor.get_metrics(timeframe_minutes)
    
    if service:
        return {
            "service": service,
            "metrics": metrics["services"].get(service, {}),
            "timestamp": datetime.now().isoformat()
        }
    
    return metrics


@router.get("/metrics/historical")
async def get_historical_metrics(
    hours: int = Query(1, ge=1, le=24),
    service: str = Query("transcription")
):
    """Get historical metrics (simplified - in production would use time-series DB)"""
    # This is a simplified version
    # In production, you'd query a time-series database
    
    metrics = monitor.get_metrics(hours * 60)  # Convert hours to minutes
    
    service_metrics = metrics["services"].get(service, {})
    
    return {
        "service": service,
        "timeframe_hours": hours,
        "samples": service_metrics.get("latency_ms", {}).get("samples", 0),
        "avg_latency_ms": service_metrics.get("latency_ms", {}).get("mean_ms", 0),
        "success_rate": service_metrics.get("success_rate", 100),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/health/detailed")
async def get_detailed_health():
    """Get detailed health status of all services"""
    metrics = monitor.get_metrics()
    
    services_status = []
    for service_name, service_metrics in metrics["services"].items():
        status = {
            "service": service_name,
            "status": service_metrics["status"],
            "active": service_metrics["active"],
            "last_activity": service_metrics["last_activity"].isoformat() 
                           if service_metrics["last_activity"] else None,
            "latency_ms": service_metrics["latency_ms"]["mean_ms"],
            "success_rate": service_metrics["success_rate"]
        }
        services_status.append(status)
    
    return {
        "system": {
            "uptime_seconds": metrics["system"]["uptime_seconds"],
            "total_requests": metrics["system"]["total_requests"],
            "active_requests": metrics["system"]["active_requests"],
            "timestamp": datetime.now().isoformat()
        },
        "services": services_status
    }