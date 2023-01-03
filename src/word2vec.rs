use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::collections::HashMap;

pub struct Word2Vec {
	pub vocab_size: u32,
	pub embed_size: u32,
	pub vector_dict: HashMap<String, Vec<f32>>
}

impl Word2Vec {
	pub fn get_vector(&self, word: &str) -> Option<&Vec<f32>> {
		self.vector_dict.get(word)
	}

	pub fn similarity(&self, word1: &str, word2: &str) -> Result<f32, &'static str> {
		if word1.eq(word2) {
			return Ok(1.0);
		}

		let word1_vector = match self.get_vector(word1) {
			Some(v) => v,
			None => return Err("Invalid first word")
		};

		let word2_vector = match self.get_vector(word2) {
			Some(v) => v,
			None => return Err("Invalid second word")
		};

		// Not normalizing the similarity output. Input vectors should already be normalized.
		let similarity = cosine_similarity(word1_vector, word2_vector);
		Ok(similarity)
	}

	pub fn get_nearest_vector(&self, search_vector: &Vec<f32>) -> Option<(String, &Vec<f32>)> {
		let mut closest_word: String = String::from("");
		let mut closest_vector: Option<&Vec<f32>> = None;
		let mut highest_similarity: f32 = 0.0;
		for (word, vector) in &self.vector_dict {
			let similarity = cosine_similarity(search_vector, vector);
			if similarity > highest_similarity {
				closest_word = String::from(word);
				closest_vector = Some(vector);
				highest_similarity = similarity;
			}
		}

		match closest_vector {
			Some(vector) => Some((closest_word, vector)),
			None => None
		}
	}

	// Actual trash
	pub fn get_nearest_vectors(&self, search_vector: &Vec<f32>, k: u32) -> Vec<(String, &Vec<f32>, f32)> {
		let mut similarities: Vec<(String, &Vec<f32>, f32)> = Vec::new();
		for (word, vector) in &self.vector_dict {
			similarities.push((word.to_string(), vector, cosine_similarity(search_vector, vector)))
		}
		similarities.sort_unstable_by(|a, b| (b.2).total_cmp(&a.2));
		similarities[0..k as usize].to_vec()
	}

	pub fn analogy(&self, from_example: &str, to_example: &str, from: &str, k: u32) -> Option<Vec<(String, &Vec<f32>, f32)>> {
		let from_example_vec = self.get_vector(from_example).unwrap();
		let to_example_vec = self.get_vector(to_example).unwrap();
		let from_vec = self.get_vector(from).unwrap();
		let analogy_vec = add_vec(&subtract_vec(to_example_vec, from_example_vec), from_vec);
		Some(self.get_nearest_vectors(&analogy_vec, k))
	}
}

pub fn add_vec(vec1: &Vec<f32>, vec2: &Vec<f32>) -> Vec<f32> {
	let mut new_vec = vec![0.0f32; vec1.len() as usize];
	for i in 0..vec1.len() as usize {
		new_vec[i] = vec1[i] + vec2[i];
	}
	new_vec
}

pub fn subtract_vec(vec1: &Vec<f32>, vec2: &Vec<f32>) -> Vec<f32> {
	let mut new_vec = vec![0.0f32; vec1.len() as usize];
	for i in 0..vec1.len() as usize {
		new_vec[i] = vec1[i] - vec2[i];
	}
	new_vec
}

pub fn scale_vec(vec: &Vec<f32>, scalar: f32) -> Vec<f32> {
	let mut new_vec = vec![0.0f32; vec.len() as usize];
	for i in 0..vec.len() as usize {
		new_vec[i] = vec[i] * scalar;
	}
	new_vec
}

pub fn normalize_vec(vec: &Vec<f32>) -> Vec<f32> {
	let mut new_vec = vec.clone();
	let mut vector_magnitude: f32 = 0.0;
	for i in 0usize..new_vec.len() as usize {
		vector_magnitude += new_vec[i].powf(2.0);
	}

	vector_magnitude = vector_magnitude.sqrt();
	for i in 0usize..new_vec.len() as usize {
		new_vec[i] /= vector_magnitude;
	}

	new_vec
}

pub fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
	let mut sum: f32 = 0.0;
	for i in 0usize..vec1.len() as usize {
		sum += vec1[i] * vec2[i];
	}
	sum
}

pub fn load_word2vec(file_path: &String) -> io::Result<Word2Vec> {
	let mut vocab_size: u32 = 0;
	let mut embed_size: u32 = 0;
	let mut vector_dict = HashMap::new();

	let f = File::open(file_path).expect("Unable to open model file");
	let f = BufReader::new(f);

	let mut counter: u32 = 0;
	for line in f.lines() {
		let safe_line = line.expect("Unable to read line");
		let input_vector: Vec<&str> = safe_line.split(' ').collect();

		if counter == 0 {
			vocab_size = input_vector[0].parse().expect("Invalid vocabulary size");
			embed_size = input_vector[1].parse().expect("Invalid embedding vector size");
		} else {
			let mut word_vector = vec![0.0f32;embed_size as usize];

			for i in 0usize..embed_size as usize {
				let dim: f32 = input_vector[i + 1].parse().expect("Invalid embedding feature");
				word_vector[i] = dim;
			}

			// Normalizing vector
			word_vector = normalize_vec(&word_vector);

			vector_dict.insert(String::from(input_vector[0]), word_vector);
		}
		counter += 1;
	}

	Ok(Word2Vec {
		vocab_size,
		embed_size,
		vector_dict
	})
}

impl std::fmt::Display for Word2Vec {
	fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
		write!(fmt, "vocab_size: {}, embed_size: {}", self.vocab_size, self.embed_size)
	}
}