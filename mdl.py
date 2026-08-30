import torch
import torch.nn as nn

with open("data.txt", "r") as file:
    text = file.read()


context_size = int(input("context size (how many words should i take in): "))
ui = input(f"enter {context_size} words to predict the next word: ").split()
wtg = int(input('how many words should i generate: '))

words = text.split()

vocab = {}

for word in words:
    if word not in vocab:
        vocab[word] = len(vocab)

tokens = []

for word in words:
    tokens.append(vocab[word])

tokens = torch.tensor(tokens)


X = []
y = []


for i in range(len(tokens) - context_size):
    X.append(tokens[i:i + context_size])
    y.append(tokens[i + context_size])

X = torch.stack(X)
y = torch.tensor(y)


class bigTransformer(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(len(vocab), 16)

        self.transformer = nn.TransformerEncoderLayer(
            d_model=16,
            nhead=2,
            batch_first=True
        )

        self.output = nn.Linear(16, len(vocab))

    def forward(self, x):

        x = self.embedding(x)

        x = self.transformer(x)

        x = x[:, -1, :]

        x = self.output(x)

        return x


model = bigTransformer()

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


for epoch in range(500):

    prediction = model(X)

    loss = loss_fn(prediction, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 50 == 0:
        print("epoch:", epoch, "loss:", loss.item())


prompt = [vocab[word] for word in ui]

for i in range(wtg):

    input_tokens = torch.tensor(prompt[-context_size:]).unsqueeze(0)

    prediction = model(input_tokens)

    next_token = prediction.argmax(dim=1).item()

    prompt.append(next_token)

for token in prompt:

    for word, number in vocab.items():

        if number == token:
            print(word, end=" ")
