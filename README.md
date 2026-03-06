# Jorkle-AI

# Simple Local LLM with JSON Knowledge Base

This project is a lightweight, local language model (LLM) built with PyTorch that learns from JSON knowledge files combined with a built-in English corpus. It supports interactive chat, dynamic learning of new facts, and easy retraining all running efficiently on a CPU.

## Features

- **Multi-Source Knowledge**: Loads and merges multiple JSON knowledge files (`knowledge.json`, `knowledge2.json`, `knowledge3.json`) to build its core intelligence.
- **Grammar Integration**: Automatically incorporates a built-in English corpus to help the model learn basic sentence structure and language flow.
- **Hybrid Response System**: Features an interactive chat mode with a smart fallback to direct JSON knowledge when the model is still in the early stages of learning.
- **On-the-Fly Learning**: Supports teaching new facts during live sessions using the `learn this X is Y` command, which updates the JSON files and a dedicated learning log.
- **CPU Optimized**: Designed to be lightweight enough to train and run on standard consumer hardware without requiring a high-end GPU.

## Requirements

- Python 3.8 or higher
- PyTorch (Install via `pip install torch`)

## Getting Started

### Training the Model

Before chatting with the AI, you must train the model. This process reads your JSON knowledge files and the built-in corpus to build the AI's "brain" (`ckpt.pt`).

Run the following command:
```bash
python LLM.py train
```
*Note: Training time depends on your CPU and the size of your JSON files, but it is optimized for small to medium datasets.*

### Chatting with the AI

Once training is complete, start an interactive chat session:
```bash
python LLM.py interactive
```
You can ask questions or have a conversation. If the AI hasn’t fully mastered a specific topic yet, it will use the fallback mechanism to provide direct answers from your JSON files.

### Teaching New Facts

You can expand the AI's knowledge during a chat session by typing:
```
learn this [Subject] is [Fact]
```
**Example:**
`learn this Python is a powerful programming language.`

This command immediately updates `knowledge.json` and appends the fact to `learned_facts.txt`. To embed these new facts deeply into the AI’s neural network, simply run the `train` command again.

## Managing Knowledge

- **Manual Updates**: You can manually edit `knowledge.json` or add new pairs to `knowledge2.json` and `knowledge3.json` at any time.
- **Grammar Base**: The `knowledge3.json` file is specifically intended for basic grammar and common phrases to help the AI form coherent sentences.
- **Retraining**: It is recommended to retrain the model periodically as your knowledge base grows to ensure the AI remains accurate and fluent.

## License

This project is released under the GNU v3 License.
