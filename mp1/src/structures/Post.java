/**
 *
 */
package structures;

import java.lang.Math;
import java.util.Map;
import java.util.HashMap;

import json.JSONException;
import json.JSONObject;

/**
 * @author hongning
 * @version 0.1
 * @category data structure
 * data structure for a Yelp review document
 * You can create some necessary data structure here to store the processed text content, e.g., bag-of-word representation
 */
public class Post {
	//unique review ID from Yelp
	String m_ID;
	public void setID(String ID) {
		m_ID = ID;
	}

	public String getID() {
		return m_ID;
	}

	//author's displayed name
	String m_author;
	public String getAuthor() {
		return m_author;
	}

	public void setAuthor(String author) {
		this.m_author = author;
	}

	//author's location
	String m_location;
	public String getLocation() {
		return m_location;
	}

	public void setLocation(String location) {
		this.m_location = location;
	}

	//review text content
	String m_content;
	public String getContent() {
		return m_content;
	}

	public void setContent(String content) {
		if (!content.isEmpty())
			this.m_content = content;
	}

	public boolean isEmpty() {
		return m_content==null || m_content.isEmpty();
	}

	//timestamp of the post
	String m_date;
	public String getDate() {
		return m_date;
	}

	public void setDate(String date) {
		this.m_date = date;
	}

	//overall rating to the business in this review
	double m_rating;
	public double getRating() {
		return m_rating;
	}

	public void setRating(double rating) {
		this.m_rating = rating;
	}

	public Post(String ID) {
		m_ID = ID;
	}

	String[] m_tokens; // we will store the tokens
	public String[] getTokens() {
		return m_tokens;
	}

	public void setTokens(String[] tokens) {
		m_tokens = tokens;
	}

	Map<String, Token> m_vector; // suggested sparse structure for storing the vector space representation with N-grams for this document
	public Map<String, Token> getVct() {
		return m_vector;
	}

	public void setVct(Map<String, Token> vct) {
		m_vector = vct;
	}


	public double similiarity(Post p) {
		Map<String, Token> p_vector = p.getVct();

		double m_norm = 0;
		double p_norm = 0;

		for (Token t: m_vector.values()) {
			m_norm += t.getValue() * t.getValue();
		}
		for (Token t: p_vector.values()) {
			p_norm += t.getValue() * t.getValue();
		}

		m_norm = Math.sqrt(m_norm);
		p_norm = Math.sqrt(p_norm);

		// when one vector is 0, cosine similarity is invalid
		if (m_norm == 0 || p_norm == 0) {
			return Double.NaN;
		}

		// it's ok to only loop one vector, coz other values are 0 anyway
		double sim = 0;
		for (Token t: m_vector.values()) {
			String key = t.getToken();
			if (!p_vector.containsKey(key)) {
				continue; // the val of key in p_vector is 0
			}
			sim += t.getValue() * p_vector.get(key).getValue();
		}

		return sim / (m_norm * p_norm);
	}

	public Post(JSONObject json) {
		try {
			m_ID = json.getString("ReviewID");
			setAuthor(json.getString("Author"));

			setDate(json.getString("Date"));
			setContent(json.getString("Content"));
			setRating(json.getDouble("Overall"));
			setLocation(json.getString("Author_Location"));
		} catch (JSONException e) {
			e.printStackTrace();
		}
	}

	public JSONObject getJSON() throws JSONException {
		JSONObject json = new JSONObject();

		json.put("ReviewID", m_ID);//must contain
		json.put("Author", m_author);//must contain
		json.put("Date", m_date);//must contain
		json.put("Content", m_content);//must contain
		json.put("Overall", m_rating);//must contain
		json.put("Author_Location", m_location);//must contain

		return json;
	}
}
