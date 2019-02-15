/**
 *
 */
package analyzer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.Math;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.englishStemmer;
import org.tartarus.snowball.ext.porterStemmer;

import json.JSONArray;
import json.JSONException;
import json.JSONObject;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InvalidFormatException;
import structures.LanguageModel;
import structures.Post;
import structures.Token;

/**
 * @author hongning
 * Sample codes for demonstrating OpenNLP package usage
 * NOTE: the code here is only for demonstration purpose,
 * please revise it accordingly to maximize your implementation's efficiency!
 */
public class DocAnalyzer {
	//N-gram to be created
	int m_N;

	//a list of stopwords
	HashSet<String> m_stopwords;

	//you can store the loaded reviews in this arraylist for further processing
	ArrayList<Post> m_reviews;

	//you might need something like this to store the counting statistics for validating Zipf's and computing IDF
	HashMap<String, Token> m_stats;

	//we have also provided a sample implementation of language model in src.structures.LanguageModel
	Tokenizer m_tokenizer;

	//this structure is for language modeling
	LanguageModel m_langModel;

	// controlled vocab with idf
	Map<String, Double> m_idf;

	public DocAnalyzer(String tokenModel, int N) throws InvalidFormatException, FileNotFoundException, IOException {
		m_N = N;
		m_reviews = new ArrayList<Post>();
		m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream(tokenModel)));
	}

	//sample code for loading a list of stopwords from file
	//you can manually modify the stopword file to include your newly selected words
	public void LoadStopwords(String filename) {
		try {
			m_stopwords = new HashSet<>();
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;

			while ((line = reader.readLine()) != null) {
				//it is very important that you perform the same processing operation to the loaded stopwords
				//otherwise it won't be matched in the text content
				line = SnowballStemming(Normalization(line));
				if (!line.isEmpty())
					m_stopwords.add(line);
			}
			// add NUM
			m_stopwords.add("NUM");
			reader.close();
			System.out.format("Loading %d stopwords from %s\n", m_stopwords.size(), filename);
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
		}
	}

	public void LoadIDF(String filename) {
		try {
			m_idf = new HashMap<>();
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;

			while ((line = reader.readLine()) != null) {
				if (!line.isEmpty()) {
					String[] parts = line.split("\t");
					m_idf.put(parts[0], Double.parseDouble(parts[2]));
				}
			}
			reader.close();
			System.out.format("Loading %d vocab from %s\n", m_idf.size(), filename);
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
		}
	}

	public void analyzeDocument(JSONObject json) {
		try {
			JSONArray jarray = json.getJSONArray("Reviews");
			for(int i=0; i<jarray.length(); i++) {
				Post review = new Post(jarray.getJSONObject(i));

				String[] tokens = Tokenize(review.getContent());

				/**
				 * HINT: perform necessary text processing here based on the tokenization results
				 * e.g., tokens -> normalization -> stemming -> N-gram -> stopword removal -> to vector
				 * The Post class has defined a "HashMap<String, Token> m_vector" field to hold the vector representation
				 * For efficiency purpose, you can accumulate a term's DF here as well
				 */
				List<String> tokenList = Arrays.asList(tokens);

				tokenList = tokenList.stream()
					.map(s -> Normalization(s))
					.map(s -> SnowballStemming(s))
					.map(s -> s.trim())
					.filter(s -> !s.isEmpty())
					.collect(Collectors.toList());

				List<String> bigrams = new ArrayList<String>();
				String prevToken = null;
				for (String token: tokenList) {
					if (prevToken != null) {
						// for bigram, add if both token are not stopwords
						if (!m_stopwords.contains(prevToken) && !m_stopwords.contains(token)) {
							bigrams.add(prevToken + "_" + token);
						}
					}
					prevToken = token;
				}

				tokenList.addAll(bigrams);

				// filter stopwords
				tokenList = tokenList.stream()
					.filter(s -> !m_stopwords.contains(s))
					.collect(Collectors.toList());

				tokens = tokenList.toArray(new String[tokenList.size()]);

				review.setTokens(tokens);

				m_reviews.add(review);
			}
		} catch (JSONException e) {
			e.printStackTrace();
		}
	}

	public void createLanguageModel() {
		m_langModel = new LanguageModel(m_N, m_stats.size());

		for(Post review:m_reviews) {
			String[] tokens = Tokenize(review.getContent());
			/**
			 * HINT: essentially you will perform very similar operations as what you have done in analyzeDocument()
			 * Now you should properly update the counts in LanguageModel structure such that we can perform maximum likelihood estimation on it
			 */
		}
	}

	//sample code for loading a json file
	public JSONObject LoadJson(String filename) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			StringBuffer buffer = new StringBuffer(1024);
			String line;

			while((line=reader.readLine())!=null) {
				buffer.append(line);
			}
			reader.close();

			return new JSONObject(buffer.toString());
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!", filename);
			e.printStackTrace();
			return null;
		} catch (JSONException e) {
			System.err.format("[Error]Failed to parse json file %s!", filename);
			e.printStackTrace();
			return null;
		}
	}

	// sample code for demonstrating how to recursively load files in a directory
	public void LoadDirectory(String folder, String suffix) {
		File dir = new File(folder);
		int size = m_reviews.size();
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix))
				analyzeDocument(LoadJson(f.getAbsolutePath()));
			else if (f.isDirectory())
				LoadDirectory(f.getAbsolutePath(), suffix);
		}
		size = m_reviews.size() - size;
		System.out.println("Loading " + size + " review documents from " + folder);
	}

	//sample code for demonstrating how to use Snowball stemmer
	public String SnowballStemming(String token) {
		SnowballStemmer stemmer = new englishStemmer();
		stemmer.setCurrent(token);
		if (stemmer.stem())
			return stemmer.getCurrent();
		else
			return token;
	}

	//sample code for demonstrating how to use Porter stemmer
	public String PorterStemming(String token) {
		porterStemmer stemmer = new porterStemmer();
		stemmer.setCurrent(token);
		if (stemmer.stem())
			return stemmer.getCurrent();
		else
			return token;
	}

	//sample code for demonstrating how to perform text normalization
	//you should implement your own normalization procedure here
	public String Normalization(String token) {
		// remove all non-word characters
		// please change this to removing all English punctuation

		// after tokenization, the actual punctuation should only appear in the either ends
		token = token.replaceAll("(^\\p{Punct}+|\\p{Punct}+$)", "");

		// convert to lower case
		token = token.toLowerCase();

		// add a line to recognize integers and doubles via regular expression

		// take care of formatted number with comma: 3,000.01
		Pattern numberPtn = Pattern.compile("\\d+(,\\d{3})*(\\.\\d+)?");
		Boolean isNumber = numberPtn.matcher(token).matches();

		// and convert the recognized integers and doubles to a special symbol "NUM"
		if (isNumber) {
			token = "NUM";
		}

		return token;
	}

	String[] Tokenize(String text) {
		return m_tokenizer.tokenize(text);
	}

	public void TokenizerDemon(String text) {
		System.out.format("Token\tNormalization\tSnonball Stemmer\tPorter Stemmer\n");
		for(String token: m_tokenizer.tokenize(text)){
			System.out.format("%s\t%s\t%s\t%s\n", token, Normalization(token), SnowballStemming(token), PorterStemming(token));
		}
	}

	private void zipfLaw() {
		LoadDirectory("./yelp/train", ".json");
		LoadDirectory("./yelp/test", ".json");

		HashMap<String, Integer> ttf = calculateTTF();
		HashMap<String, Integer> df = calculateDF();

		System.out.println("[ttf]");
		sortPrintMap(ttf);

		System.out.println("[df]");
		sortPrintMap(df);
	}

	private HashMap<String, Integer> calculateTTF() {
		HashMap<String, Integer> ttf = new HashMap<>();

		for (Post review: m_reviews) {
			for (String token: review.getTokens()) {
				if (ttf.containsKey(token)) {
					ttf.put(token, ttf.get(token) + 1);
				} else {
					ttf.put(token, 1);
				}
			}
		}

		return ttf;
	}

	private HashMap<String, Integer> calculateDF() {
		HashMap<String, Integer> df = new HashMap<>();

		for (Post review: m_reviews) {
			Set<String> tokenSet = new HashSet<>(Arrays.asList(review.getTokens()));
			for (String token: tokenSet) {
				if (df.containsKey(token)) {
					df.put(token, df.get(token) + 1);
				} else {
					df.put(token, 1);
				}
			}
		}

		return df;
	}

	private void sortPrintMap(Map<String, Integer> map) {
		List<Entry<String, Integer>> list = new ArrayList<>(map.entrySet());
    list.sort(Entry.<String, Integer>comparingByValue().reversed());

		for (Entry<String, Integer> entry: list) {
			System.out.println(entry.getKey() + "\t" + entry.getValue());
		}
	}

	private void specificStopWords(Map<String, Integer> df) {
		List<Entry<String, Integer>> list = new ArrayList<>(df.entrySet());
		list.sort(Entry.<String, Integer>comparingByValue().reversed());

		List<Entry<String, Integer>> top100 = list.subList(0, 100);
		LoadStopwords("./data/stop_words.txt");
		for (Entry<String, Integer> entry: top100) {
			String token = entry.getKey();
			if (!m_stopwords.contains(token)) {
				System.out.println(token);
			}
		}
	}

	private void controlledVocab() {
		LoadStopwords("./data/stop_words.txt");
		LoadDirectory("./yelp/train", ".json");
		Map<String, Integer> df = calculateDF();

		m_idf = df.entrySet()
			.stream()
			.filter(e -> e.getValue() >= 50)
			.collect(Collectors.toMap(
				Entry::getKey,
				e -> 1 + Math.log((double) m_reviews.size() / e.getValue())
			));

		List<Entry<String, Double>> list = new ArrayList<>(m_idf.entrySet());
		list.sort(Entry.<String, Double>comparingByValue());

		for (Entry<String, Double> entry: list) {
			System.out.println(entry.getKey() + "\t" + df.get(entry.getKey()) + "\t" + entry.getValue());
		}

		// specificStopWords(df);
	}

	private void setVector(Post review) {
		Map<String, Integer> counts = new HashMap<>();
		for (String token: review.getTokens()) {
			// not in vocab
			if (!m_idf.containsKey(token)) {
				continue;
			}

			if (counts.containsKey(token)) {
				counts.put(token, counts.get(token) + 1);
			} else {
				counts.put(token, 1);
			}
		}

		Map<String, Double> tf = counts.entrySet().stream()
			.collect(Collectors.toMap(
				Entry::getKey,
				e -> 1 + Math.log(e.getValue())
			));

		Map<String, Double> weights =	tf.entrySet().stream()
			.collect(Collectors.toMap(
				Entry::getKey,
				e -> e.getValue() * m_idf.get(e.getKey())
			));

		Map<String, Token> vct = new HashMap<String, Token>();
		for (Entry<String, Double> entry:  weights.entrySet()) {
			Token t = new Token(entry.getKey());
			t.setValue(entry.getValue());
			vct.put(entry.getKey(), t);
		}

		review.setVct(vct);
	}

	private void computeSim() {
		LoadStopwords("./data/stop_words.txt");
		LoadIDF("./data/vocab.txt");

		LoadDirectory("./yelp/query", ".json");
		List<Post> q_reviews = m_reviews;
		for (Post review: q_reviews) {
			setVector(review);
		}

		m_reviews = new ArrayList<>();
		LoadDirectory("./yelp/test", ".json");
		for (Post review: m_reviews) {
			setVector(review);
		}

		for (Post q_review: q_reviews) {
			Map<Post, Double> results = new HashMap<>();
			for (Post m_review: m_reviews) {
				Double sim = q_review.similiarity(m_review);
				if (!sim.isNaN()) {
					results.put(m_review, sim);
				}
			}

			List<Entry<Post, Double>> list = new ArrayList<>(results.entrySet());
			list.sort(Entry.<Post, Double>comparingByValue().reversed());

			System.out.println("Top 3 for query: " + q_review.getID());
			System.out.println(q_review.getContent());
			System.out.println("\n");

			for (Entry<Post, Double> entry: list.subList(0, 3)) {
				Post review = entry.getKey();
				System.out.println(review.getID() + "\t" + review.getAuthor() + "\t" + review.getDate() +  "\t" + entry.getValue());
				System.out.println(review.getContent());
			}

			System.out.println("\n\n");
		}
	}

	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException {
		DocAnalyzer analyzer = new DocAnalyzer("./data/Model/en-token.bin", 2);

		//code for demonstrating tokenization and stemming
		// analyzer.TokenizerDemon("I've don't practiced for 30 30,000 30.1 002 2.2.2 3.3a 3,0,1,0.001 years in pediatrics, I've never seen anything quite like this.");

		//entry point to deal with a collection of documents
		// analyzer.zipfLaw();

		// analyzer.controlledVocab();

		analyzer.computeSim();
	}

}
