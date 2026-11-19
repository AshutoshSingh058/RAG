from typing import List
import logfire

def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 100) -> List[str]:
    """
    Simple semantic-ish chunker that splits by paragraphs.
    Ensures chunks do not exceed the specified size while preserving a small
    amount of context between adjacent chunks.
    """
    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        if not text.strip(): 
            return []
            
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue

            if len(current_chunk) + len(p) < chunk_size:
                current_chunk += p + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = p + "\n\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        valid_chunks = [c for c in chunks if c.strip()]
        if overlap:
            overlapped_chunks = [valid_chunks[0]] if valid_chunks else []
            for previous, chunk in zip(valid_chunks, valid_chunks[1:]):
                prefix_size = min(overlap, max(0, chunk_size - len(chunk) - 2))
                prefix = previous[-prefix_size:].strip() if prefix_size else ""
                overlapped_chunks.append(
                    f"{prefix}\n\n{chunk}".strip() if prefix else chunk
                )
            valid_chunks = overlapped_chunks
        logfire.info(f"✅ Generated {len(valid_chunks)} chunks")
        return valid_chunks
