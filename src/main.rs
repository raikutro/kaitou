use std::fs::File;
use std::io::{self, BufRead, BufReader};
use kaitou::word2vec::load_word2vec;
use kaitou::model::Kaitou;
use kaitou::model::KaitouConfig;

fn main() {
	let args: Vec<String> = std::env::args().collect();

	let mut model = Kaitou::create(KaitouConfig {
		w2v_path: String::from(&args[1]),
		tokenizer_path: String::from(&args[2]),
		frequency_path: String::from(&args[3]),
		analysis_folder_path: None,

		analogy_top_k: 20,
		semantic_similarity_minimum: 0.35,
		distance_idf_cross_convergence: None,
		scalar_func: Some(|x, y| {
			1.0
		}),
		semantic_similarity_matrix_modifier_func: Some(|x| {
			*x
		})
	}).unwrap();

	let training_file = String::from(&args[4]);
	let f = File::open(training_file).expect("Unable to open corpus file");
	let f = BufReader::new(f);

	for line in f.lines() {
		let safe_line = line.expect("Unable to read line");
		let pair: Vec<&str> = safe_line.split("|||").collect();
		model.train(
			&String::from(pair[0]),
			&String::from(pair[1])
		).unwrap();
	}

	loop {
		let mut prompt = String::new();

		io::stdin()
			.read_line(&mut prompt)
			.expect("Failed to read line");

		if prompt.starts_with(&String::from("R:")) {
			println!("{}", model.respond(&String::from(&prompt[2..])).unwrap());
		} else if prompt.starts_with(&String::from("T:")) {
			let training_exchange = String::from(&prompt[2..]);
			let pair: Vec<&str> = training_exchange.split("|||").collect();
			model.train(
				&String::from(pair[0]),
				&String::from(pair[1])
			).unwrap();
		} else {
			println!("{}", "ERR: Invalid Request");
		}
	}
}