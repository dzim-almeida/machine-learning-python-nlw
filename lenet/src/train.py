import torch
import torch.optim as optim
from src.model import LeNet
from src.dataset import get_mnist_data, get_data_loaders

def train_model(epochs=10, learning_rate=0.001, batch_size=64, data_dir='./data'):
    """
    Trains the LeNet model on the MNIST dataset.
    """
    # Get data
    train_dataset, test_dataset = get_mnist_data(data_dir)
    train_loader, test_loader = get_data_loaders(train_dataset, test_dataset, batch_size)

    # Initialize model, loss, and optimizer
    model = LeNet()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 200 == 199:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0
    
    print('Finished Training')
    
    # Save the trained model
    torch.save(model.state_dict(), 'lenet_mnist.pth')
    print('Model saved to lenet_mnist.pth')

    # Evaluate the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
