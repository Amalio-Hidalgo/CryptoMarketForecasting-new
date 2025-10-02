"""
Dask Helper Utilities for Cryptocurrency Volatility Forecasting

This module contains utilities for Dask cluster management and optimization.
"""

import os
from typing import Optional, Dict, Any
from dask.distributed import Client, LocalCluster


def create_optimized_dask_client(
    n_workers: int = 4,
    threads_per_worker: int = 5,
    memory_limit: str = '8GB',
    dashboard_port: int = 8787,
    processes: bool = True,
    silence_logs: bool = True
) -> Client:
    """
    Create an optimized Dask client for crypto volatility forecasting.
    
    Args:
        n_workers: Number of worker processes
        threads_per_worker: Threads per worker
        memory_limit: Memory limit per worker
        dashboard_port: Port for Dask dashboard
        processes: Use processes instead of threads
        silence_logs: Reduce logging verbosity
        
    Returns:
        Configured Dask client
    """
    try:
        # Close existing client/cluster if any
        try:
            client = Client.current()
            client.close()
        except ValueError:
            pass
            
        # Create optimized cluster
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            processes=processes,
            dashboard_address=f':{dashboard_port}',
            memory_limit=memory_limit,
            silence_logs=silence_logs
        )
        
        client = Client(cluster)
        
        print(f"Dask client created successfully!")
        print(f"   Workers: {n_workers}")
        print(f"   Threads per worker: {threads_per_worker}")
        print(f"   Memory per worker: {memory_limit}")
        print(f"   Dashboard: http://localhost:{dashboard_port}")
        
        return client
        
    except Exception as e:
        print(f"Error creating Dask client: {e}")
        raise


def create_coiled_cluster(
    n_workers: int = 3,
    worker_memory: str = "32GiB",
    scheduler_memory: str = "8GiB",
    region: str = "eu-west-3",
    shutdown_on_close: bool = True
) -> Optional[Client]:
    """
    Create AWS Coiled cluster for large-scale processing.
    
    Args:
        n_workers: Number of workers
        worker_memory: Memory per worker
        scheduler_memory: Scheduler memory
        region: AWS region
        shutdown_on_close: Shutdown cluster when closed
        
    Returns:
        Coiled cluster client
    """
    try:
        import coiled
        
        cluster = coiled.Cluster(
            n_workers=n_workers,
            worker_memory=worker_memory,
            scheduler_memory=scheduler_memory,
            region=region,
            shutdown_on_close=shutdown_on_close,
        )
        
        client = cluster.get_client()
        
        print(f"☁️ Coiled cluster created successfully!")
        print(f"   Workers: {n_workers}")
        print(f"   Worker memory: {worker_memory}")
        print(f"   Region: {region}")
        print(f"   Dashboard: {client.dashboard_link}")
        
        return client
        
    except ImportError:
        print("Coiled not available. Install with: pip install coiled")
        return None
    except Exception as e:
        print(f"Error creating Coiled cluster: {e}")
        return None


def get_dask_config() -> Dict[str, Any]:
    """
    Get Dask configuration from environment variables.
    
    Returns:
        Dictionary with Dask configuration
    """
    return {
        'n_workers': int(os.getenv('DASK_WORKERS', 4)),
        'threads_per_worker': int(os.getenv('DASK_THREADS_PER_WORKER', 5)),
        'memory_limit': os.getenv('DASK_MEMORY_LIMIT', '8GB'),
        'dashboard_port': int(os.getenv('DASK_DASHBOARD_PORT', 8787))
    }


def cleanup_dask_client(client: Optional[Client] = None) -> None:
    """
    Clean up Dask client and cluster.
    
    Args:
        client: Dask client to clean up (uses current if None)
    """
    try:
        if client is None:
            client = Client.current()
            
        if hasattr(client, 'cluster'):
            client.cluster.close()
        client.close()
        
        print("Dask client cleaned up successfully")
        
    except ValueError:
        print("No active Dask client to clean up")
    except Exception as e:
        print(f"Error cleaning up Dask client: {e}")


if __name__ == "__main__":
    # Test Dask utilities
    print("Testing Dask utilities...")
    
    client = create_optimized_dask_client(n_workers=2, memory_limit='4GB')
    print(f"Client created: {client}")
    
    cleanup_dask_client(client)
    print("✅ Dask utilities test completed!")