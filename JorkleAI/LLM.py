import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse

knowledge_files = ["knowledge.json", "knowledge2.json", "knowledge3.json"]
learned_facts_file = "learned_facts.txt"
brain_file = "ckpt.pt"

def get_base_text():
    return (
        "Alice was beginning to get very tired of sitting by her sister on the bank, "
        "and of having nothing to do: once or twice she had peeped into the book her sister was reading, "
        "but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice "
        "'without pictures or conversation?' "
        "So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), "
        "whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, "
        "when suddenly a White Rabbit with pink eyes ran close by her."
    )

def prepare_data():
    full_text = ""
    for f_name in knowledge_files:
        if os.path.exists(f_name):
            try:
                with open(f_name, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        if isinstance(v, list):
                            for item in v:
                                full_text += f"{k} is {item}. "
                        else:
                            full_text += f"{k} is {v}. "
            except:
                continue
    
    if os.path.exists(learned_facts_file):
        try:
            with open(learned_facts_file, "r", encoding="utf-8") as f:
                full_text += "\n" + f.read()
        except:
            pass
            
    full_text += "\n" + get_base_text()
    
    if len(full_text) < 50:
        full_text += " The AI is learning from JSON and text. " * 10
        
    return full_text

class TextVocab:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
    def encode(self, s): 
        return [self.stoi.get(c, 0) for c in s]
    def decode(self, ids): 
        return "".join(self.chars[i] for i in ids)
    def __len__(self): 
        return len(self.chars)

class TextData(Dataset):
    def __init__(self, encoded, size):
        self.data = torch.tensor(encoded, dtype=torch.long)
        self.size = size
    def __len__(self): 
        return len(self.data) - self.size
    def __getitem__(self, i):
        return self.data[i:i+self.size], self.data[i+1:i+1+self.size]

class BrainModel(nn.Module):
    def __init__(self, v_size, n_embd=128, b_size=64):
        super().__init__()
        self.tok_emb = nn.Embedding(v_size, n_embd)
        self.pos_emb = nn.Embedding(b_size, n_embd)
        self.blocks = nn.Sequential(*[nn.TransformerEncoderLayer(d_model=n_embd, nhead=4, batch_first=True) for _ in range(2)])
        self.out = nn.Linear(n_embd, v_size)
        self.b_size = b_size
        
    def forward(self, x):
        b, t = x.size()
        p = torch.arange(0, t, device=x.device).unsqueeze(0)
        x = self.tok_emb(x) + self.pos_emb(p)
        x = self.blocks(x)
        return self.out(x)
        
    @torch.no_grad()
    def generate(self, x, count=50):
        for _ in range(count):
            x_cond = x[:, -self.b_size:]
            logits = self(x_cond)[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_id), dim=1)
        return x

def run_training():
    text = prepare_data()
    vocab = TextVocab(text)
    encoded = vocab.encode(text)
    dataset = TextData(encoded, 64)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = BrainModel(len(vocab))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(5):
        for i, (x, y) in enumerate(loader):
            logits = model(x)
            loss = loss_fn(logits.view(-1, len(vocab)), y.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % 5 == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(loader)} | Loss: {loss.item():.4f}")
                
    save_data = {'model': model.state_dict(), 'vocab': vocab.chars}
    torch.save(save_data, brain_file)
    print(f"Done! Brain saved to {brain_file}")

def start_chat():
    if not os.path.exists(brain_file):
        print("No brain found. Run 'python LLM.py train' first.")
        return
        
    save_data = torch.load(brain_file)
    vocab = TextVocab("".join(save_data['vocab']))
    vocab.chars = save_data['vocab']
    vocab.stoi = {c: i for i, c in enumerate(vocab.chars)}
    
    model = BrainModel(len(vocab))
    model.load_state_dict(save_data['model'])
    model.eval()
    
    kb = {}
    for f_name in knowledge_files:
        if os.path.exists(f_name):
            try:
                with open(f_name, "r") as f:
                    kb.update(json.load(f))
            except:
                pass
                
    print("\nAI Ready. Type 'exit' to stop.")
    while True:
        user_in = input("You: ").strip()
        if user_in.lower() == 'exit': 
            break
            
        if user_in.lower().startswith("learn this "):
            stuff = user_in[11:]
            if " is " in stuff:
                k, v = stuff.split(" is ", 1)
                k, v = k.strip(), v.strip()
                
                current_kb = {}
                if os.path.exists("knowledge.json"):
                    with open("knowledge.json", "r") as f:
                        current_kb = json.load(f)
                current_kb[k] = v
                
                with open("knowledge.json", "w") as f:
                    json.dump(current_kb, f, indent=2)
                    
                with open(learned_facts_file, "a", encoding="utf-8") as f:
                    f.write(f"{k} is {v}.\n")
                    
                print(f"Saved '{k}' to knowledge.json. Run 'train' to update the brain.")
                continue
        
        ans = None
        for key in kb:
            if key.lower() in user_in.lower():
                ans = kb[key]
                break
        
        if ans:
            print(f"AI: {ans}")
            continue
            
        ctx = torch.tensor([vocab.encode(user_in)], dtype=torch.long)
        gen = model.generate(ctx)
        res = vocab.decode(gen[0].tolist()[len(user_in):]).strip()
        
        if len(set(res[:10])) < 3:
            print("AI: I'm still learning how to speak. Please teach me more!")
        else:
            print(f"AI: {res.split('.')[0]}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["train", "interactive"])
    args = parser.parse_args()
    
    if args.cmd == "train":
        run_training()
    else:
        start_chat()