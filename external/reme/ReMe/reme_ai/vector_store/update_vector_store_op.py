"""Operation for updating the vector store with memory changes."""
import asyncio
import json
from typing import List, Dict

from flowllm.core.context import C
from flowllm.core.op import BaseAsyncOp
from flowllm.core.schema import VectorNode
from loguru import logger

from reme_ai.schema.memory import BaseMemory


class ReadWriteLock:
    """A read-write lock implementation for asyncio.
    
    Allows multiple concurrent readers or a single exclusive writer.
    Readers can proceed concurrently, but writers are exclusive.
    """
    
    def __init__(self):
        self._readers = 0
        self._writer = False
        self._lock = asyncio.Lock()
        self._read_condition = asyncio.Condition(self._lock)
        self._write_condition = asyncio.Condition(self._lock)
    
    async def acquire_read(self):
        """Acquire a read lock. Multiple readers can hold this simultaneously."""
        async with self._lock:
            # Wait until no writer is active
            while self._writer:
                await self._read_condition.wait()
            self._readers += 1
    
    async def release_read(self):
        """Release a read lock."""
        async with self._lock:
            self._readers -= 1
            if self._readers == 0:
                # Notify waiting writers
                self._write_condition.notify_all()
    
    async def acquire_write(self):
        """Acquire a write lock. Exclusive - no readers or other writers allowed."""
        async with self._lock:
            # Wait until no readers and no writer
            while self._readers > 0 or self._writer:
                await self._write_condition.wait()
            self._writer = True
    
    async def release_write(self):
        """Release a write lock."""
        async with self._lock:
            self._writer = False
            # Notify waiting readers and writers
            self._read_condition.notify_all()
            self._write_condition.notify_all()
    
    async def __aenter__(self):
        """Async context manager entry for read lock."""
        await self.acquire_read()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit for read lock."""
        await self.release_read()
    
    class ReadLock:
        """Context manager for read lock."""
        def __init__(self, rw_lock: 'ReadWriteLock'):
            self.rw_lock = rw_lock
        
        async def __aenter__(self):
            await self.rw_lock.acquire_read()
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self.rw_lock.release_read()
    
    class WriteLock:
        """Context manager for write lock."""
        def __init__(self, rw_lock: 'ReadWriteLock'):
            self.rw_lock = rw_lock
        
        async def __aenter__(self):
            await self.rw_lock.acquire_write()
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self.rw_lock.release_write()
    
    def read_lock(self):
        """Get a read lock context manager."""
        return self.ReadLock(self)
    
    def write_lock(self):
        """Get a write lock context manager."""
        return self.WriteLock(self)


_workspace_locks: Dict[str, ReadWriteLock] = {}
_locks_lock: asyncio.Lock = None

def _get_locks_lock() -> asyncio.Lock:
    global _locks_lock
    if _locks_lock is None:
        _locks_lock = asyncio.Lock()
    return _locks_lock


async def get_workspace_rw_lock(workspace_id: str) -> ReadWriteLock:
    """Get or create a read-write lock for a workspace.
    
    This function is async-safe and can be called from multiple coroutines.
    
    Args:
        workspace_id: The workspace ID to get the lock for.
        
    Returns:
        The ReadWriteLock instance for the workspace.
    """
    locks_lock = _get_locks_lock()
    async with locks_lock:
        if workspace_id not in _workspace_locks:
            _workspace_locks[workspace_id] = ReadWriteLock()
        return _workspace_locks[workspace_id]



@C.register_op()
class UpdateVectorStoreOp(BaseAsyncOp):
    """Operation that synchronizes memory changes with the vector store.

    This operation performs the actual database operations: deleting old
    memories and inserting new/updated ones. It reads the lists of memories
    to delete and insert from the context response metadata.
    """
    
    async def _get_workspace_write_lock(self, workspace_id: str) -> ReadWriteLock.WriteLock:
        """Get a write lock for the workspace.
        
        This method uses the read-write lock mechanism to ensure exclusive
        access for write operations (insert/delete).
        """
        rw_lock = await get_workspace_rw_lock(workspace_id)
        return rw_lock.write_lock()

    async def async_execute(self):
        """Execute the vector store update operation.

        Performs batch operations on the vector store:
        1. Deletes memories specified in deleted_memory_ids
        2. Inserts new or updated memories from memory_list

        The operation reads from response.metadata and performs the actual
        database operations. Results are stored back in response.metadata.

        Expected context attributes:
            workspace_id: The workspace ID to update.

        Expected response.metadata:
            deleted_memory_ids: List of memory IDs to delete from vector store.
            memory_list: List of BaseMemory objects to insert into vector store.

        Sets context attributes:
            response.metadata["update_result"]: Dictionary with deletion and
                insertion counts.
        """
        workspace_id: str = self.context.workspace_id

        write_lock = await self._get_workspace_write_lock(workspace_id)

        async with write_lock:
            deleted_memory_ids: List[str] = self.context.response.metadata.get("deleted_memory_ids", [])
            if deleted_memory_ids:
                await self.vector_store.async_delete(node_ids=deleted_memory_ids, workspace_id=workspace_id)
                logger.info(f"delete memory_ids={json.dumps(deleted_memory_ids, indent=2)}")

            insert_memory_list: List[BaseMemory] = self.context.response.metadata.get("memory_list", [])
            if insert_memory_list:
                insert_nodes: List[VectorNode] = [x.to_vector_node() for x in insert_memory_list]
                await self.vector_store.async_insert(nodes=insert_nodes, workspace_id=workspace_id)
                logger.info(f"insert insert_node.size={len(insert_nodes)}")

            # Store results in context
            self.context.response.metadata["update_result"] = {
                "deleted_count": len(deleted_memory_ids) if deleted_memory_ids else 0,
                "inserted_count": len(insert_memory_list) if insert_memory_list else 0,
            }
