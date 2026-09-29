#include "analyzer.hpp"
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char* argv[]) {
    std::string text_to_analyze;

    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if ((arg == "--text" || arg == "-t") && i + 1 < argc) {
                text_to_analyze = argv[++i];
            } else if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: persona_cpp_analyzer [--text \"Text to analyze\"]" << std::endl;
                std::cout << "Reads from stdin if --text is not provided." << std::endl;
                return 0;
            } else if (text_to_analyze.empty() && arg[0] != '-') {
                text_to_analyze = arg;
            }
        }
    }

    if (text_to_analyze.empty()) {
        std::stringstream buffer;
        buffer << std::cin.rdbuf();
        text_to_analyze = buffer.str();
    }

    persona::TextAnalyzer analyzer;
    persona::TextAnalysisResult result = analyzer.analyze(text_to_analyze);
    std::cout << result.to_json() << std::endl;

    return 0;
}
