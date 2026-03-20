use pyo3_stub_gen::Result;

fn main() -> Result<()> {
	let stub = synaplex::stub_info()?;
	stub.generate()?;
	Ok(())
}
