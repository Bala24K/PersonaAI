#include "analyzer.hpp"
#include <cassert>
#include <iostream>

int main() {
    persona::TextAnalyzer analyzer;

    std::string sample = "I am so happy and excited today! Everything is working great.";
    auto res = analyzer.analyze(sample);

    assert(res.char_count > 0);
    assert(res.word_count > 5);
    assert(res.emotion_keyword_counts.at("positive") >= 2);
    assert(res.feature_hash_16.size() == 16);

    std::string json = res.to_json();
    assert(json.find("\"word_count\":") != std::string::npos);

    std::cout << "All C++ analyzer tests passed successfully!" << std::endl;
    return 0;
}
