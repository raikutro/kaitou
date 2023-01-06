use std::fs::File;
use std::io::{self, BufRead, BufReader};
use kaitou::word2vec::load_word2vec;
use kaitou::model::Kaitou;
use kaitou::model::KaitouConfig;

fn main() {
	println!("{}", std::env::current_dir().unwrap().display());
	// word2vec_test();

	let mut model = Kaitou::create(KaitouConfig {
		w2v_path: String::from("../../training/models/wiki_300_skipgram/w2v.txt"),
		tokenizer_path: String::from("../../training/models/wiki_300_skipgram/tokens.json"),
		frequency_path: String::from("../../training/models/wiki_300_skipgram/frequency.txt"),
		analysis_folder_path: Some(String::from("../analysis")),

		analogy_top_k: 20,
		semantic_similarity_minimum: 0.25,
		distance_idf_cross_convergence: None,
		scalar_func: Some(|x, y| {
			1.0
		}),
		semantic_similarity_matrix_modifier_func: Some(|x| {
			*x
		})
	}).unwrap();

	let training_file = String::from("../../extraction/clean/caring.txt");
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

	model.train(
		&String::from("What is your favorite color?"),
		&String::from("My favorite color is white.")
	).unwrap();
	model.train(
		&String::from("What is your favorite food?"),
		&String::from("I like pizza.")
	).unwrap();
	model.train(
		&String::from("your favorite bird?"),
		&String::from("falcons are cool.")
	).unwrap();

	// println!("{:?}", model.exchanges);

	let questions = vec![
		String::from("Hey"),
		String::from("My girlfriend left me..."),
		String::from("I'm sad."),
		String::from("I'm back from work"),
		String::from("It sucked")
	];

	for question in questions {
		println!("Q: {}, A: {}", question, model.respond(&question).unwrap());
	}
}

fn word2vec_test() {
	let w2v_result = load_word2vec(&String::from("../example_data/w2v_model.txt"));

	let w2v = match w2v_result {
		Ok(model) => model,
		Err(e) => panic!("Problem opening the model file: {:?}", e)
	};

	println!("Model: {} | First Vector: {:?}", w2v, w2v.get_vector("the"));
	println!("Similarity: {:?}", w2v.similarity("the", "i"));

	// let test_vector = match w2v.get_vector("the") {
	// 	Some(v) => v,
	// 	None => panic!("Invalid vector")
	// };
	// println!("Closest vectors: {:?}", w2v.get_nearest_vectors(test_vector, 10));
	println!("Analogy: {:?}", w2v.analogy("suisei", "hololive", "delta", 10));
}
