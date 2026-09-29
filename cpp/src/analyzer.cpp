#include "analyzer.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <set>

namespace persona {

static std::string to_lower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return result;
}

static std::string escape_json(const std::string& s) {
    std::ostringstream o;
    for (char c : s) {
        switch (c) {
            case '"': o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\b': o << "\\b"; break;
            case '\f': o << "\\f"; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default:
                if ('\x00' <= c && c <= '\x1f') {
                    o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
                } else {
                    o << c;
                }
        }
    }
    return o.str();
}

TextAnalyzer::TextAnalyzer() {
    emotion_lexicon_["positive"] = {"happy", "great", "awesome", "good", "love", "thanks", "helpful", "wonderful", "excited", "nice"};
    emotion_lexicon_["negative"] = {"sad", "bad", "terrible", "awful", "hate", "angry", "upset", "depressed", "miserable", "worst"};
    emotion_lexicon_["anxiety"] = {"anxious", "worried", "nervous", "scared", "afraid", "panic", "stressed", "overwhelmed", "dread"};
    emotion_lexicon_["urgency"] = {"now", "urgent", "asap", "immediately", "quick", "fast", "deadline", "emergency", "hurry"};
}

uint32_t TextAnalyzer::hash_string(const std::string& str) const {
    // FNV-1a 32-bit hash
    uint32_t hash = 2166136261u;
    for (char c : str) {
        hash ^= static_cast<uint8_t>(c);
        hash *= 16777619u;
    }
    return hash;
}

size_t TextAnalyzer::count_syllables(const std::string& word) const {
    std::string w = to_lower(word);
    if (w.empty()) return 0;
    
    size_t count = 0;
    bool prev_is_vowel = false;
    static const std::string vowels = "aeiouy";

    for (char c : w) {
        bool is_vowel = (vowels.find(c) != std::string::npos);
        if (is_vowel && !prev_is_vowel) {
            count++;
        }
        prev_is_vowel = is_vowel;
    }

    if (w.length() > 2 && w.back() == 'e' && w[w.length() - 2] != 'l') {
        if (count > 1) count--;
    }

    return count == 0 ? 1 : count;
}

TextAnalysisResult TextAnalyzer::analyze(const std::string& text) const {
    TextAnalysisResult res;
    res.char_count = text.length();
    res.feature_hash_16.assign(16, 0.0);

    if (text.empty()) {
        return res;
    }

    // Tokenize words and sentences
    std::vector<std::string> words;
    size_t sentence_cnt = 0;
    std::string current_word;
    size_t total_syllables = 0;

    for (size_t i = 0; i < text.length(); ++i) {
        char c = text[i];
        if (c == '.' || c == '!' || c == '?') {
            if (i == 0 || (text[i - 1] != '.' && text[i - 1] != '!' && text[i - 1] != '?')) {
                sentence_cnt++;
            }
        }

        if (std::isalnum(static_cast<unsigned char>(c))) {
            current_word += c;
        } else {
            if (!current_word.empty()) {
                words.push_back(current_word);
                total_syllables += count_syllables(current_word);
                current_word.clear();
            }
        }
    }
    if (!current_word.empty()) {
        words.push_back(current_word);
        total_syllables += count_syllables(current_word);
    }

    if (sentence_cnt == 0) sentence_cnt = 1;
    res.word_count = words.size();
    res.sentence_count = sentence_cnt;

    if (res.word_count == 0) {
        return res;
    }

    // Lexical statistics
    size_t total_word_len = 0;
    std::set<std::string> unique_words;
    for (const auto& w : words) {
        std::string lower_w = to_lower(w);
        unique_words.insert(lower_w);
        total_word_len += w.length();

        // 16-dim feature hashing projection
        uint32_t h = hash_string(lower_w);
        size_t bucket = h % 16;
        double sign = ((h >> 16) & 1) ? 1.0 : -1.0;
        res.feature_hash_16[bucket] += sign;
    }

    res.avg_word_length = static_cast<double>(total_word_len) / res.word_count;
    res.type_token_ratio = static_cast<double>(unique_words.size()) / res.word_count;

    // Flesch Reading Ease estimate
    double words_per_sentence = static_cast<double>(res.word_count) / res.sentence_count;
    double syllables_per_word = static_cast<double>(total_syllables) / res.word_count;
    res.readability_score = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word);
    if (res.readability_score < 0.0) res.readability_score = 0.0;
    if (res.readability_score > 100.0) res.readability_score = 100.0;

    // Normalize feature hash vector
    double norm = 0.0;
    for (double val : res.feature_hash_16) {
        norm += val * val;
    }
    norm = std::sqrt(norm);
    if (norm > 1e-6) {
        for (double& val : res.feature_hash_16) {
            val /= norm;
        }
    }

    // Emotion keyword scanning
    for (const auto& pair : emotion_lexicon_) {
        const std::string& category = pair.first;
        const std::vector<std::string>& lex_words = pair.second;
        int count = 0;
        for (const auto& w : words) {
            std::string lw = to_lower(w);
            for (const auto& target : lex_words) {
                if (lw == target) {
                    count++;
                }
            }
        }
        res.emotion_keyword_counts[category] = count;
    }

    return res;
}

std::string TextAnalysisResult::to_json() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "{\n";
    ss << "  \"char_count\": " << char_count << ",\n";
    ss << "  \"word_count\": " << word_count << ",\n";
    ss << "  \"sentence_count\": " << sentence_count << ",\n";
    ss << "  \"avg_word_length\": " << avg_word_length << ",\n";
    ss << "  \"type_token_ratio\": " << type_token_ratio << ",\n";
    ss << "  \"readability_score\": " << readability_score << ",\n";
    ss << "  \"emotion_keyword_counts\": {\n";

    size_t idx = 0;
    for (const auto& pair : emotion_keyword_counts) {
        const std::string& k = pair.first;
        int v = pair.second;
        ss << "    \"" << escape_json(k) << "\": " << v;
        if (++idx < emotion_keyword_counts.size()) ss << ",";
        ss << "\n";
    }
    ss << "  },\n";

    ss << "  \"feature_hash_16\": [";
    for (size_t i = 0; i < feature_hash_16.size(); ++i) {
        ss << feature_hash_16[i];
        if (i + 1 < feature_hash_16.size()) ss << ", ";
    }
    ss << "]\n";
    ss << "}";

    return ss.str();
}

} // namespace persona
