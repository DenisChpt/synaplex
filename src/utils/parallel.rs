/// Compute an optimal chunk size for parallel iteration.
/// Aims for at least `min_chunks` chunks to keep all cores busy,
/// with a minimum chunk size of `min_size` to avoid overhead.
pub fn adaptive_chunk_size(total: usize, min_size: usize, min_chunks: usize) -> usize {
	let num_threads = rayon::current_num_threads();
	let target_chunks = num_threads * min_chunks;
	let chunk = total / target_chunks.max(1);
	chunk.max(min_size)
}
