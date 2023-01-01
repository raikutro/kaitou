use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
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
	pub fn get_nearest_vectors(&self, search_vector: &Vec<f32>, k: i32) -> Vec<(String, &Vec<f32>, f32)> {
		let mut distances: Vec<(String, &Vec<f32>, f32)> = Vec::new();
		for (word, vector) in &self.vector_dict {
			distances.push((word.to_string(), vector, cosine_similarity(search_vector, vector)))
		}
		distances.sort_unstable_by(|a, b| (b.2).total_cmp(&a.2));
		distances[0..k as usize].to_vec()
	}

	pub fn analogy(&self, from_example: &str, to_example: &str, from: &str, k: i32) -> Option<Vec<(String, &Vec<f32>, f32)>> {
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

pub fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
	let mut sum: f32 = 0.0;
	for i in 0usize..vec1.len() as usize {
		sum += vec1[i] * vec2[i];
	}
	sum
}

pub fn load_word2vec(file_path: &str) -> io::Result<Word2Vec> {
	let mut vocab_size: u32 = 0;
	let mut embed_size: u32 = 0;
	let mut vector_dict = HashMap::new();

	match read_lines(file_path) {
		Ok(lines) => {
			let mut counter: u32 = 0;
			for line in lines {
				if let Ok(safe_line) = line {
					let input_vector: Vec<&str> = safe_line.split(' ').collect();

					if counter == 0 {
						vocab_size = input_vector[0].parse().expect("Invalid vocabulary size");
						embed_size = input_vector[1].parse().expect("Invalid embedding vector size");
					} else {
						let mut word_vector = vec![0.0f32;embed_size as usize];

						// Decompressing vectors
						let mut squared_sum: f32 = 0.0;
						for i in 0usize..embed_size as usize {
							let dim: f32 = input_vector[i + 1].parse().expect("Invalid embedding feature");
							word_vector[i] = dim;
							squared_sum += dim.powf(2.0);
						}

						let sqroot_sum = squared_sum.sqrt();
						for i in 0usize..embed_size as usize {
							word_vector[i] /= sqroot_sum;
						}

						vector_dict.insert(String::from(input_vector[0]), word_vector);
					}
					counter += 1;
				}
			}
		},
		Err(e) => panic!("Problem opening the model file: {:?}", e),
	};

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

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
	let file = File::open(filename)?;
	Ok(io::BufReader::new(file).lines())
}