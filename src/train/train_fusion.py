import torch.optim as optim
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fusion_model = FusionModel(cnn_model, text_model, num_classes=2)
fusion_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(fusion_model.fc.parameters(), lr=1e-4)  # train fusion layers only

epochs = 5

for epoch in range(epochs):
    fusion_model.train()
    total_loss = 0
    for images, input_ids, attention_mask, labels in train_loader:
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = fusion_model(images, input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

fusion_model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, input_ids, attention_mask, labels in test_loader:
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = fusion_model(images, input_ids, attention_mask)
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {correct/total:.4f}")
