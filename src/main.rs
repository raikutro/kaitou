use kaitou::word2vec::load_word2vec;

fn main() {
	println!("{}", std::env::current_dir().unwrap().display());
	let w2v_result = load_word2vec("../example_data/w2v_model.txt");

	let w2v = match w2v_result {
		Ok(model) => model,
		Err(e) => panic!("Problem opening the model file: {:?}", e)
	};

	println!("Model: {} | First Vector: {:?}", w2v, w2v.get_vector("the"));
	println!("Similarity: {:?}", w2v.similarity("the", "i"));

	let test_vector = match w2v.get_vector("the") {
		Some(v) => v,
		None => panic!("Invalid vector")
	};

	// println!("Closest vectors: {:?}", w2v.get_nearest_vectors(test_vector, 10));
	println!("Analogy: {:?}", w2v.analogy("suisei", "hololive", "delta", 10));
}
