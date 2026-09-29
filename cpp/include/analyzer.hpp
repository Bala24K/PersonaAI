#ifndef PERSONA_ANALYZER_HPP
#define PERSONA_ANALYZER_HPP

#include <string>
#include <vector>
#include <map>
#include <cstdint>

namespace persona {

struct TextAnalysisResult {
    size_t char_count{0};
    size_t word_count{0};
    size_t sentence_count{0};
    double avg_word_length{0.0};
    double type_token_ratio{0.0};
    double readability_score{0.0};
    std::map<std::string, int> emotion_keyword_counts;
    std::vector<double> feature_hash_16;

    std::string to_json() const;
};

class TextAnalyzer {
public:
    TextAnalyzer();
    TextAnalysisResult analyze(const std::string& text) const;

private:
    std::map<std::string, std::vector<std::string>> emotion_lexicon_;
    size_t count_syllables(const std::string& word) const;
    uint32_t hash_string(const std::string& str) const;
};

} // namespace persona

#endif // PERSONA_ANALYZER_HPP
