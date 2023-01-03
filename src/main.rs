use kaitou::word2vec::load_word2vec;
use kaitou::model::Kaitou;
use kaitou::model::KaitouConfig;

fn main() {
	println!("{}", std::env::current_dir().unwrap().display());
	// word2vec_test();

	let mut model = Kaitou::create(KaitouConfig {
		w2v_path: String::from("../../training/models/wiki_200_skipgram/w2v.txt"),
		tokenizer_path: String::from("../../training/models/wiki_200_skipgram/tokens.json"),
		frequency_path: String::from("../../training/models/wiki_200_skipgram/frequency.txt"),
		analysis_folder_path: Some(String::from("../analysis")),

		analogy_top_k: 10,
		semantic_similarity_minimum: 0.35,
		scalar_func: Some(|x, y| {
			// This equation was derived by analyzing the model dump analytics and performing
			// a multiple linear regression on the similarity and IDF score -> scalar.
			(-2.6153+(9.5273*x)+(46.8842*y)-(6.9402*x*x)-(20.8846*x*y)-(99.2212*y*y))
				.max(0.0).min(1.0) * 1.5 // Clamp the values
		})
	}).unwrap();

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

	println!("{:?}", model.exchanges);

	let questions = vec![
		String::from("What is your favorite animal?"),
		String::from("What is your favorite food?"),
		String::from("your favorite bird?"),
		String::from("your favorite color?"),
		String::from("What is the best animal?"),
		String::from("What's your favorite drink?"),
		String::from("What's your favorite mood?"),
		String::from("What's your favorite lunch meal?"),
		String::from("Are pancakes your favorite food?"),
		String::from("What's better than your favorite color?"),
		String::from("Are you sentient?"),
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
