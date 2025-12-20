"""Memory deduplication operation for task memory management.

This module provides operations to remove duplicate or highly similar task
memories by comparing embeddings and calculating similarity scores.
"""

import asyncio
from typing import List

from flowllm.core.context import C
from flowllm.core.op import BaseAsyncOp
from loguru import logger

from reme_ai.schema.memory import BaseMemory
from reme_ai.vector_store.update_vector_store_op import get_workspace_rw_lock


@C.register_op()
class MemoryDeduplicationOp(BaseAsyncOp):
    """Remove duplicate task memories using embedding similarity.

    This operation identifies and removes duplicate or highly similar task
    memories by comparing their embeddings against both existing memories
    in the vector store and other memories in the current batch.
    """

    file_path: str = __file__

    async def async_execute(self):
        """Remove duplicate task memories"""
        # Get task memories to deduplicate
        task_memories: List[BaseMemory] = self.context.response.metadata.get("memory_list", [])

        if not task_memories:
            logger.info("No task memories found for deduplication")
            return

        logger.info(f"Starting deduplication for {len(task_memories)} task memories")

        # Perform deduplication
        deduplicated_task_memories = await self._deduplicate_task_memories(task_memories)

        logger.info(
            f"Deduplication complete: {len(deduplicated_task_memories)} deduplicated "
            f"task memories out of {len(task_memories)}",
        )

        # Update context
        self.context.response.metadata["memory_list"] = deduplicated_task_memories

    async def _deduplicate_task_memories(self, task_memories: List[BaseMemory]) -> List[BaseMemory]:
        """Remove duplicate task memories"""
        if not task_memories:
            return task_memories

        similarity_threshold = self.op_params.get("similarity_threshold", 0.5)
        workspace_id = self.context.get("workspace_id")

        unique_task_memories = []

        # Get existing task memory embeddings
        existing_embeddings = await self._get_existing_task_memory_embeddings(workspace_id)

        for task_memory in task_memories:
            # Generate embedding for current task memory
            current_embedding = self._get_task_memory_embedding(task_memory)

            if current_embedding is None:
                logger.warning(f"Failed to generate embedding for task memory: {str(task_memory.when_to_use)[:50]}...")
                continue

            # Check similarity with existing task memories
            if self._is_similar_to_existing_task_memories(current_embedding, existing_embeddings, similarity_threshold):
                logger.debug(f"Skipping similar task memory: {str(task_memory.when_to_use)[:50]}...")
                continue

            # Check similarity with current batch task memories
            if self._is_similar_to_current_task_memories(current_embedding, unique_task_memories, similarity_threshold):
                logger.debug(f"Skipping duplicate in current batch: {str(task_memory.when_to_use)[:50]}...")
                continue

            # Add to unique task memories list
            unique_task_memories.append(task_memory)
            logger.debug(f"Added unique task memory: {str(task_memory.when_to_use)[:50]}...")

        return unique_task_memories
        

    async def _get_existing_task_memory_embeddings(self, workspace_id: str) -> List[List[float]]:
        """Get embeddings of existing task memories.
        
        Uses read lock to allow concurrent reads while preventing conflicts
        with write operations. Includes retry mechanism for transient errors.
        """
        if not hasattr(self, "vector_store") or not self.vector_store or not workspace_id:
            return []

        # Retry configuration
        max_retries = 3
        retry_delay = 0.5  # seconds
        base_delay = 0.1  # base delay for exponential backoff
        
        # Get read lock for the workspace
        rw_lock = await get_workspace_rw_lock(workspace_id)
        read_lock = rw_lock.read_lock()
        
        for attempt in range(max_retries):
            try:
                async with read_lock:
                    # Query existing task memory nodes
                    existing_nodes = await self.vector_store.async_search(
                        query="...",  # Empty query to get all
                        workspace_id=workspace_id,
                        top_k=self.op_params.get("max_existing_task_memories", 1000),
                    )

                    # Extract embeddings
                    existing_embeddings = []
                    for node in existing_nodes:
                        if hasattr(node, "embedding") and node.embedding:
                            existing_embeddings.append(node.embedding)

                    logger.debug(
                        f"Retrieved {len(existing_embeddings)} existing task memory embeddings from workspace {workspace_id}",
                    )
                    return existing_embeddings

            except (OSError, IOError) as e:
                # Check if it's the "Resource temporarily unavailable" error
                # error_code = getattr(e, 'errno', None)
                # if error_code == 11 or "Resource temporarily unavailable" in str(e):
                    if attempt < max_retries - 1:
                        # Exponential backoff with jitter
                        delay = base_delay * (2 ** attempt) + retry_delay
                        logger.warning(
                            f"Resource temporarily unavailable when retrieving embeddings "
                            f"(attempt {attempt + 1}/{max_retries}), retrying in {delay:.2f}s: {e}"
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.warning(
                            f"Failed to retrieve existing task memory embeddings after {max_retries} attempts: {e}"
                        )
                        return []
            
            except Exception as e:
                # Other exceptions - log and return empty
                logger.warning(f"Failed to retrieve existing task memory embeddings: {e}")
                return []
        
        # Should not reach here, but return empty list as fallback
        return []

    def _get_task_memory_embedding(self, task_memory: BaseMemory) -> List[float] | None:
        """Generate embedding for task memory"""
        try:

            # Combine task memory description and content for embedding
            text_for_embedding = f"{task_memory.when_to_use} {task_memory.content}"
            embeddings = self.vector_store.embedding_model.get_embeddings([text_for_embedding])

            if embeddings and len(embeddings) > 0:
                return embeddings[0]
            else:
                logger.warning("Empty embedding generated for task memory")
                return None

        except Exception as e:
            logger.error(f"Error generating embedding for task memory: {e}")
            return None

    def _is_similar_to_existing_task_memories(
        self,
        current_embedding: List[float],
        existing_embeddings: List[List[float]],
        threshold: float,
    ) -> bool:
        """Check if current embedding is similar to existing embeddings"""
        for existing_embedding in existing_embeddings:
            similarity = self._calculate_cosine_similarity(current_embedding, existing_embedding)
            if similarity > threshold:
                logger.debug(f"Found similar existing task memory with similarity: {similarity:.3f}")
                return True
        return False

    def _is_similar_to_current_task_memories(
        self,
        current_embedding: List[float],
        current_task_memories: List[BaseMemory],
        threshold: float,
    ) -> bool:
        """Check if current embedding is similar to other memories in current batch."""
        for existing_task_memory in current_task_memories:
            existing_embedding = self._get_task_memory_embedding(existing_task_memory)
            if existing_embedding is None:
                continue

            similarity = self._calculate_cosine_similarity(current_embedding, existing_embedding)
            if similarity > threshold:
                logger.debug(f"Found similar task memory in current batch with similarity: {similarity:.3f}")
                return True
        return False

    @staticmethod
    def _calculate_cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity"""
        try:
            import numpy as np

            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)

        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
