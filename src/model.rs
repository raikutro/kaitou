use std::fs::File;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader};
use std::io::Write;
use std::collections::HashMap;
use std::time::{SystemTime};
use rand::Rng;

use crate::word2vec::Word2Vec;
use crate::word2vec::load_word2vec;
use crate::word2vec::{add_vec, subtract_vec, scale_vec, normalize_vec};

use tokenizers::tokenizer::Tokenizer;
use tokenizers::tokenizer::Result as TokenResult;

use levenshtein_diff as levenshtein;
use levenshtein_diff::edit::Edit;

pub struct Kaitou {
	pub model_id: String,
	pub w2v: Word2Vec,
	pub exchanges: Vec<SequenceExchange>,
	pub frequency_ranks: HashMap<u32, u32>,
	pub total_tokens: u32,
	pub tokenizer: Tokenizer,
	pub config: KaitouConfig,
	pub analysis_response_writer: Option<File>,
}

type ScalarCalcFn = fn(&f32, &f32) -> f32;
type SemanticSimilarityModifierFn = fn(&f32) -> f32;

pub struct KaitouConfig {
	// Model files
	pub w2v_path: String,
	pub tokenizer_path: String,
	pub frequency_path: String,

	// Optional folder path to dump data for analyzing the model's performance
	pub analysis_folder_path: Option<String>,

	// Language Model Settings
	pub analogy_top_k: u32,
	pub semantic_similarity_minimum: f32,
	pub distance_idf_cross_convergence: Option<f32>,
	pub scalar_func: Option<ScalarCalcFn>,
	pub semantic_similarity_matrix_modifier_func: Option<SemanticSimilarityModifierFn>,
}

impl Kaitou {
	pub fn create(config: KaitouConfig) -> TokenResult<Self> {
		let model_id = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs().to_string();
		let w2v_model = load_word2vec(&config.w2v_path).unwrap();
		let tokenizer = Tokenizer::from_file(config.tokenizer_path.as_str())?;
		let mut frequency_ranks: HashMap<u32, u32> = HashMap::new();
		let mut total_tokens = 0;

		let mut freq_vec: Vec<(u32, u32)> = vec![];

		// Parse frequency file
		let f = File::open(config.frequency_path.as_str()).expect("Unable to open frequency file");
		let f = BufReader::new(f);
		let mut counter: u32 = 0;
		for line in f.lines() {
			let safe_line = line.expect("Unable to read line");
			let entry: Vec<&str> = safe_line.split(' ').collect();

			if counter == 0 {
				total_tokens = entry[0].parse().expect("Invalid total tokens");
			} else {
				freq_vec.push((
					entry[0].parse().expect("Invalid token"),
					entry[1].parse().expect("Invalid frequency value")
				));
			}
			counter += 1;
		}

		// Store frequency rankings
		freq_vec.sort_unstable_by(|a, b| b.1.cmp(&a.1));
		for i in 0..freq_vec.len() {
			frequency_ranks.insert(freq_vec[i].0, i as u32);
		}

		let mut analysis_response_writer: Option<File> = None;
		if let Some(analysis_folder_path) = &config.analysis_folder_path {
			analysis_response_writer = Some(OpenOptions::new()
				.append(true)
				.create(true)
				.open(format!("{}/response_dump_{}.txt", analysis_folder_path, model_id))
				.expect("Unable to create response dump file"));
		}

		Ok(Self {
			model_id,
			w2v: w2v_model,
			exchanges: vec![],
			frequency_ranks,
			total_tokens,
			tokenizer,
			config,
			analysis_response_writer
		})
	}

	pub fn train(&mut self, input: &String, output: &String) -> TokenResult<u32> {
		let input_tokens = self.tokenizer.encode(input.as_str(), false)?;
		let output_tokens = self.tokenizer.encode(output.as_str(), false)?;
		// println!("{:?} {:?}", input_tokens.get_ids(), output_tokens.get_ids());
		let sequence_id = self.exchanges.len();

		self.exchanges.push(
			SequenceExchange::new(
				(input_tokens.get_ids().to_vec(), output_tokens.get_ids().to_vec()),
				&self.w2v,
				&self
			)
		);

		Ok(sequence_id as u32)
	}

	pub fn respond(&mut self, input: &String) -> TokenResult<String> {
		let input_tokens = self.tokenizer.encode(input.as_str(), false)?.get_ids().to_vec();
		let mut distances: Vec<(usize, usize)> = self.exchanges
			.iter()
			.enumerate()
			.map(|(idx, ex)| (levenshtein::distance(&ex.input, &input_tokens).0, idx)).collect();
		distances.sort_unstable_by(|a, b| (a.0 as u32).cmp(&(b.0 as u32)));
		// println!("{:?}", distances);

		let chosen_distances: Vec<(usize, usize)> = distances[0..2].to_vec();
		let chosen_index = rand::thread_rng().gen_range(0..=2);

		let exchange_index = distances[chosen_index].1;
		let template_in_seq = &self.exchanges[exchange_index].input;
		let (_, leven_matrix) = levenshtein::distance(&template_in_seq, &input_tokens);
		let levenshtein_edits = levenshtein::generate_edits(&template_in_seq, &input_tokens, &leven_matrix).unwrap();
		let response_template = &self.exchanges[exchange_index].output;

		let mut response_tokens = response_template.clone();

		// println!("{:?}", &self.exchanges[exchange_index]);

		for edit in levenshtein_edits {
			let change: (usize, u32) = match edit {
				// Edit indexes are 1-based. Normalizing to 0-based.
				Edit::Substitute(idx, val) => (idx - 1, val),
				Edit::Insert(idx, val) => continue,
				Edit::Delete(_) => continue
			};

			// println!("Change: {:?}", change);

			for i in 0..response_template.len() {
				// Preparing features
				// SCS
				let semantic_connection_strength = &self.exchanges[exchange_index].matrix_get(change.0 as u32, i as u32);
				// RT-IDF
				let response_template_token_inverse_frequency_score = self.inverse_frequency_score(response_template[i]);

				// If the semantic connection between the changed token and the response token are
				// less than the minimum, skip the token.
				if semantic_connection_strength < &self.config.semantic_similarity_minimum {
					continue;
				}

				// Analogy Equation
				// Oo + (In - Io)λ = On
				let old_output_token = self.w2v.get_vector(&(response_template[i].to_string())).unwrap();
				let new_input_token = self.w2v.get_vector(&(change.1.to_string())).unwrap();
				let old_input_token = self.w2v.get_vector(&(template_in_seq[change.0].to_string())).unwrap();

				// If the model was provided a scalar function, use it.
				let scalar = match &self.config.scalar_func {
					Some(func) => func(semantic_connection_strength, &response_template_token_inverse_frequency_score),
					None => semantic_connection_strength + response_template_token_inverse_frequency_score
				};

				// If the scalar function fails, skip this token.
				if scalar == 0.0 {
					continue;
				}

				// Apply the scale, calculate the analogy, then find the k-nearest closest tokens.
				let scaled_displacement_vector = scale_vec(&subtract_vec(&new_input_token, &old_input_token), scalar);
				let analogy_vec = normalize_vec(&add_vec(&old_output_token, &scaled_displacement_vector));
				let mut closest_vectors: Vec<(String, &Vec<f32>, f32)> = self.w2v.get_nearest_vectors(&analogy_vec, self.config.analogy_top_k);

				if (
					!self.tokenizer.id_to_token(response_template[i]).unwrap().chars().all(char::is_alphanumeric) ||
					!self.tokenizer.id_to_token(template_in_seq[change.0]).unwrap().chars().all(char::is_alphanumeric)
				) {
					continue;
				}

				// If the top analogy's similarity is less than .8, find a better word.
				let mut new_output_token = closest_vectors[0].0.clone();
				if closest_vectors[0].2 < 0.8 {
					// Filter out all of the tokens that were used to create the analogy (Oo, In, Io)
					closest_vectors.retain(|vec| {
						let token_id: u32 = vec.0.parse().unwrap();
						!(
							&token_id == &response_template[i] ||
							&token_id == &change.1 ||
							&token_id == &template_in_seq[change.0]
						)
					});

					// Skip this token if there are no vectors
					if closest_vectors.len() == 0 {
						continue;
					}

					// Calculate a cross feature between each tokens idf and distance to the analogy vector
					// Also calculates the mean cross value
					
					// --- Normalize values ---
					
					let initial_similarity = closest_vectors[0].2;
					let initial_idf_score = self.inverse_frequency_score(closest_vectors[0].0.parse().unwrap());

					// Calculate min and max
					let min_token_similarity = closest_vectors.iter().fold(initial_similarity, |acc, e| acc.min(e.2));
					let max_token_similarity = closest_vectors.iter().fold(initial_similarity, |acc, e| acc.max(e.2));
					let min_token_idf = closest_vectors.iter().fold(
						initial_idf_score,
						|acc, e| acc.min(self.inverse_frequency_score(e.0.parse().unwrap()))
					);
					let max_token_idf = closest_vectors.iter().fold(
						initial_idf_score,
						|acc, e| acc.max(self.inverse_frequency_score(e.0.parse().unwrap()))
					);

					let mut normalized_closest_vectors = vec![];
					for (token, _, similarity) in &closest_vectors {
						normalized_closest_vectors.push((
							token,
							map_range((min_token_similarity, max_token_similarity), (0.0, 1.0), *similarity),
							map_range((min_token_idf, max_token_idf), (0.0, 1.0), self.inverse_frequency_score(token.parse().unwrap()))
						));
					}

					let mut mean_cross = 0.0;
					for (token, idf, similarity) in &normalized_closest_vectors {
						mean_cross += similarity * idf;
						// println!("{:?}: {:?}", self.tokenizer.id_to_token(token.parse().unwrap()), similarity * idf);
					}
					mean_cross = match self.config.distance_idf_cross_convergence {
						Some(n) => n,
						None => mean_cross / closest_vectors.len() as f32
					};
					// println!("IDF-Distance Mean Cross: {:?}", mean_cross);

					// Find the token that is closest to the IDF-Distance Mean Cross feature.
					let mut lowest_deviation_token = new_output_token.clone();
					let mut lowest_deviation = 1.0;

					for (token, idf, similarity) in &normalized_closest_vectors {
						let deviation = (mean_cross - (similarity * idf)).abs();
						if deviation < lowest_deviation {
							// if !self.tokenizer.id_to_token(token).contains(".") && !self.tokenizer.id_to_token(token).contains("'") {
								lowest_deviation = deviation;
								lowest_deviation_token = token.to_string();
							// }
						}
					}

					new_output_token = lowest_deviation_token.clone();
				}

				let new_output_token: u32 = new_output_token.parse().unwrap();
				// println!(
				// 	"{:?} + ({:?} - {:?}){:?} = {:?}",
				// 	self.tokenizer.id_to_token(response_template[i]).unwrap(),
				// 	self.tokenizer.id_to_token(change.1).unwrap(),
				// 	self.tokenizer.id_to_token(template_in_seq[change.0]).unwrap(),
				// 	scalar,
				// 	self.tokenizer.id_to_token(new_output_token).unwrap()
				// );

				// Write data to the analysis dump if it exists
				if let Some(file) = &mut self.analysis_response_writer {
					writeln!(file, "{:?},{:?},{:?},{:?},{:?},{:?},{:?}",
						semantic_connection_strength,
						response_template_token_inverse_frequency_score,
						scalar,
						self.tokenizer.id_to_token(response_template[i]).unwrap(),
						self.tokenizer.id_to_token(new_output_token).unwrap(),
						input,
						self.tokenizer.decode(response_tokens.clone(), true),
					).expect("Couldn't write to analysis file");
				}

				response_tokens[i] = new_output_token;
			}
		}

		let response_string: String = self.tokenizer.decode(response_tokens, true).unwrap();

		Ok(response_string)
	}
	pub fn frequency_score(&self, token: u32) -> f32 {
		1.0 - self.inverse_frequency_score(token)
	}

	pub fn inverse_frequency_score(&self, token: u32) -> f32 {
		(*self.frequency_ranks.get(&token).unwrap() as f32) / (self.frequency_ranks.len() as f32)
	}
}

pub struct SequenceExchange {
	input: Vec<u32>,
	output: Vec<u32>,
	semantic_matrix: Vec<f32>,
	matrix_width: u32
}

impl SequenceExchange {
	pub fn new(sequence: (Vec<u32>, Vec<u32>), w2v: &Word2Vec, model: &Kaitou) -> Self {
		let matrix_width: u32 = sequence.0.len() as u32;
		let matrix_size: usize = sequence.0.len() * sequence.1.len();
		let mut exchange = Self {
			input: sequence.0,
			output: sequence.1,
			semantic_matrix: vec![0.0; matrix_size],
			matrix_width
		};

		for i in 0..exchange.input.len() {
			for j in 0..exchange.output.len() {
				let mut similarity = w2v.similarity(
					&(exchange.input[i].to_string()),
					&(exchange.output[j].to_string())
				).unwrap_or(0.0);

				similarity = match &model.config.semantic_similarity_matrix_modifier_func {
					Some(func) => func(&similarity),
					None => similarity
				};

				exchange.matrix_set(i as u32, j as u32, similarity);
			}
		}

		exchange
	}

	pub fn matrix_get(&self, x: u32, y: u32) -> f32 {
		self.semantic_matrix[(x + self.matrix_width * y) as usize]
	}

	pub fn matrix_set(&mut self, x: u32, y: u32, v: f32) {
		self.semantic_matrix[(x + self.matrix_width * y) as usize] = v;
	}
}

fn round_prec(x: f32, decimals: u32) -> f32 {
	let y = 10i32.pow(decimals) as f32;
	(x * y).floor() / y
}

fn map_range(from_range: (f32, f32), to_range: (f32, f32), s: f32) -> f32 {
    to_range.0 + (s - from_range.0) * (to_range.1 - to_range.0) / (from_range.1 - from_range.0)
}

impl std::fmt::Debug for SequenceExchange {
	fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
		let mut matrix_display_string = String::from("    ");

		for i in 0..self.semantic_matrix.len() {
			matrix_display_string.push_str(&(format!("{: >5}", round_prec(self.semantic_matrix[i], 2).to_string())));
			matrix_display_string.push(' ');
			if (i + 1) % (self.matrix_width as usize) == 0 && i != self.semantic_matrix.len() - 1 {
				matrix_display_string.push_str("\n    ");
			}
		}

		write!(fmt, "\n{}\n", matrix_display_string)
	}
}